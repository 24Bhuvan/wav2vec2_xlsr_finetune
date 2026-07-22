import json
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Union
import torch

# Set up logging to align with the core pipeline structure
logger = logging.getLogger("inference")


def align_single_file(
    audio_path: Union[str, Path],
    transcript_path: Union[str, Path],
    emission_generator: Any,
    align_tokens_fn: Callable[[Any, List[int]], Union[List[Dict[str, Any]], None]],
    merge_tokens_fn: Callable[[List[Dict[str, Any]], List[int], Any, float], List[Dict[str, Any]]],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Performs end-to-end forced alignment on a single audio-transcript pair.

    Args:
        audio_path: Path to the target raw audio file (.wav).
        transcript_path: Path to the target text transcript file.
        emission_generator: Initialized wrapper instance from emission.py.
        align_tokens_fn: The manual CTC path matching function from decoder.py.
        merge_tokens_fn: The word reconstruction boundary engine from postprocess.py.
        output_dir: Destination directory for exporting alignment JSONs.

    Returns:
        dict: Performance tracking metadata and runtime status properties.
    """
    audio_path = Path(audio_path)
    transcript_path = Path(transcript_path)
    output_dir = Path(output_dir)

    base_name = audio_path.stem
    output_json_path = output_dir / f"{base_name}.json"

    stats: Dict[str, Any] = {
        "file_id": base_name,
        "status": "Failed",
        "audio_duration_sec": 0.0,
        "num_words_aligned": 0,
        "error_message": None,
    }

    try:
        # 1. Transcript verification and reading
        if not transcript_path.is_file():
            raise FileNotFoundError(f"Transcript missing at target location: {transcript_path}")

        transcript = transcript_path.read_text(encoding="utf-8").strip().upper()

        if not transcript:
            raise ValueError(f"Transcript target at {transcript_path} is completely empty.")

        # 2. Extract Acoustic Feature Frame Matrix
        # Unpacks according to emission.py definition: (emissions, frame_duration)
        emissions, frame_duration = emission_generator.generate_emissions(str(audio_path))

        # ==================================================
        # DIAGNOSTIC SECTION: GREEDY CTC DECODE
        # ==================================================
        pred_ids = torch.argmax(emissions, dim=-1).unsqueeze(0)
        prediction = emission_generator.processor.batch_decode(pred_ids)[0]
        
        print("==================================================")
        print("GREEDY CTC DECODE")
        print("==================================================")
        print(f"Prediction : {prediction}")
        print(f"Target     : {transcript}")
        # ==================================================

        # 3. Compute frame spatial metrics
        num_frames = emissions.shape[0]
        duration = num_frames * frame_duration
        stats["audio_duration_sec"] = round(duration, 2)

        # 4. Tokenization Prep and Character Space Alignment
        # IMPORTANT: We must NOT use tokenizer.encode() here.
        # tokenizer.encode() is a full NLP pipeline that injects special tokens
        # (e.g., <s>, </s>) and does NOT guarantee a 1-to-1 character→token_id mapping.
        # For CTC forced alignment, every character in the transcript must map to
        # exactly one token ID so that the Viterbi trellis token_index values are
        # consistent with the actual character positions in the transcript.
        # The correct approach is to split into characters first, then look each up.
        tokenizer = emission_generator.processor.tokenizer
        cleaned_transcript = transcript.replace(" ", "|")
        chars = list(cleaned_transcript)
        token_ids = tokenizer.convert_tokens_to_ids(chars)

        # 5. Execute Mathematical Trellis Space Resolution
        alignment_points = align_tokens_fn(emissions, token_ids)

        print("\n" + "=" * 70)
        print(f"FILE                : {audio_path.name}")
        print(f"TRANSCRIPT          : {transcript}")
        print(f"TRANSCRIPT CHARS   : {len(token_ids)}")
        print(f"ALIGNMENT POINTS   : {len(alignment_points) if alignment_points else 0}")

        if alignment_points:
            valid_points = [p for p in alignment_points if p["token_index"] >= 0]
            print(f"VALID TOKEN POINTS : {len(valid_points)}")

            print("\nFIRST 30 VALID ALIGNMENT POINTS")
            for point in valid_points[:30]:
                print(point)

        if alignment_points is None:
            raise RuntimeError(
                "CTC alignment mapping calculations failed: Trellis grid graph disconnected."
            )

        # 6. Transform Frame Offsets to Timeline Arrays
        # Conforms strictly to merge_tokens_to_words parameters without extra variables
        word_timestamps = merge_tokens_fn(alignment_points, token_ids, tokenizer, frame_duration)

        print(f"\nRECOVERED WORDS : {len(word_timestamps)}")

        for word in word_timestamps:
            print(word)

        # 7. Write Result Parameters out to JSON Artifacts
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(word_timestamps, indent=4), encoding="utf-8")

        stats["status"] = "Success"
        stats["num_words_aligned"] = len(word_timestamps)
        logger.info("Successfully generated timestamps artifact: %s", output_json_path)

    except Exception as exc:
        stats["status"] = "Failed"
        # Extract full track error maps instead of basic strings for production profiling
        stats["error_message"] = traceback.format_exc()
        logger.error("Processing sequence exception context: \n%s", stats["error_message"])

    return stats
