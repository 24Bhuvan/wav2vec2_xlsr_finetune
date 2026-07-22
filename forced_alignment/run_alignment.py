import os
import glob
import logging
import traceback
from pathlib import Path
from utils import (
    setup_pipeline_logger, 
    get_optimal_device, 
    validate_path,
    normalize_transcript
)
from emission import EmissionGenerator
from decoder import align_tokens
from postprocess import merge_tokens_to_words
from align import align_single_file
from compare import compare_annotations
from evaluate import run_evaluation

# ---------------------------------------------------------
# Project Paths (Resolves execution from project root)
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
AUDIO_INPUT_DIR = DATA_DIR / "audio"
TRANSCRIPT_INPUT_DIR = DATA_DIR / "transcripts"
HUMAN_ANNOTATION_DIR = DATA_DIR / "annotations"  # Fixed folder name target

OUTPUT_DIR = PROJECT_ROOT / "forced_alignment" / "outputs"
PREDICTED_OUTPUT_DIR = OUTPUT_DIR / "predicted_annotations"

MODEL_DIR = PROJECT_ROOT / "inference_model"

def run_pipeline():
    """Executes the full end-to-end alignment, comparison, and evaluation stack."""
    # 1. Initialize core diagnostic logging infrastructure
    logger = setup_pipeline_logger()
    logger.info("Starting complete forced alignment workflow initialization.")
    
    print("=" * 60)
    print("Initializing Forced Alignment Execution Pipeline")
    print("=" * 60)

    # 2. Path Validation checks (Converted to string for compatibility with utility)
    if not validate_path(str(AUDIO_INPUT_DIR), is_directory=True):
        error_msg = f"Audio source path directory missing: {AUDIO_INPUT_DIR}"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        return

    # Create target directories on the fly if needed
    validate_path(str(PREDICTED_OUTPUT_DIR), is_directory=True, create_if_missing=True)

    # 3. Discover data files (Updated to look for .mp3 based on your dataset)
    audio_files = sorted(glob.glob(str(AUDIO_INPUT_DIR / "*.mp3")))
    total_files = len(audio_files)
    
    if total_files == 0:
        logger.warning(f"No valid .mp3 files found inside: {AUDIO_INPUT_DIR}")
        print(f"No audio files found inside {AUDIO_INPUT_DIR}. Exiting pipeline.")
        return

    logger.info(f"Discovered {total_files} audio source files to process.")
    print(f"Discovered {total_files} target files for alignment.")
    print("Loading optimized acoustic model weights...")

    # 4. Instantiate isolated model runtime dependencies (EXPOSES EXCEPTION DETAILED BELOW)
    print(f"Initializing acoustic inference pipeline from: {MODEL_DIR}")
    try:
        # Pass path as a string to the generator dependency
        generator = EmissionGenerator(model_dir=str(MODEL_DIR))
    except Exception as e:
        print("\n" + "!" * 60)
        print("CRITICAL MODEL LOADING FAILURE DETECTED")
        print("!" * 60)
        traceback.print_exc()  # Direct stack trace delivery to standard output
        logger.exception("Failed to initialize acoustic model")
        raise  # Immediate termination preventing silent errors downstream

    success_count = 0
    failure_count = 0

    print("\nRunning alignment sequences...")
    print("-" * 60)

    # 5. Core sequential processing loop
    for idx, audio_path in enumerate(audio_files, start=1):
        # Correctly isolate base name regardless of extension
        base_name = Path(audio_path).stem
        transcript_path = str(TRANSCRIPT_INPUT_DIR / f"{base_name}.txt")
        
        # Simple inline text progress tracker display
        progress_pct = (idx / total_files) * 100
        print(f"[{idx}/{total_files}] {progress_pct:6.2f}% | Aligning: {base_name}...", end="\r", flush=True)

        # Trigger batch execution helper step
        stats = align_single_file(
            audio_path=audio_path,
            transcript_path=transcript_path,
            emission_generator=generator,
            align_tokens_fn=align_tokens,
            merge_tokens_fn=merge_tokens_to_words,
            output_dir=str(PREDICTED_OUTPUT_DIR)
        )

        # Log localized operational data outcomes
        if stats["status"] == "Success":
            success_count += 1
            logger.info(f"Successfully aligned [{base_name}] - {stats['num_words_aligned']} words.")
        else:
            failure_count += 1
            logger.error(f"Failed to align [{base_name}] | Reason: {stats['error_message']}")

    print("\n" + "-" * 60)
    print(f"Alignment batch pass complete. Success: {success_count} | Failed: {failure_count}")
    logger.info(f"Batch generation sequence finished. Success: {success_count}, Failed: {failure_count}")

    # 6. Post-processing analytics orchestration steps
    print("\nRunning error comparison against ground-truth datasets...")
    logger.info("Executing downstream analytical comparison scripts.")
    try:
        compare_annotations(
            predicted_dir=str(PREDICTED_OUTPUT_DIR),
            human_dir=str(HUMAN_ANNOTATION_DIR),
            output_dir=str(OUTPUT_DIR),
            threshold_ms=70.0
        )
    except Exception as e:
        logger.error(f"Comparison report calculation failure: {str(e)}")
        print(f"Error executing comparison phase: {str(e)}")

    print("Compiling data telemetry dashboard figures...")
    logger.info("Executing downstream global telemetry metric evaluation modules.")
    print("\n" + "=" * 50)
    
    try:
        # Run analytical verification matrices reporting out to standard display blocks
        run_evaluation(threshold_ms=70.0)
    except Exception as e:
        logger.error(f"Evaluation report aggregation runtime crash: {str(e)}")
        print(f"Error executing global validation dashboard step: {str(e)}")
        print("=" * 50)

    logger.info("Pipeline execution lifecycle ended cleanly.")

if __name__ == "__main__":
    run_pipeline()
