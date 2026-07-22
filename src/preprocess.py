import os
import re
import json
import hashlib
import logging
import random
import shutil
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf


import pandas as pd
import torch
import torchaudio

# ==============================================================================
# CONFIGURATION - ADJUST THESE PATHS TO POINT TO YOUR RAW DATASET LOCATION
# ==============================================================================
RAW_AUDIO_DIR = "data/raw/audio"
RAW_TRANSCRIPT_DIR = "data/raw/transcripts"
RAW_ANNOTATION_DIR = "data/raw/annotations"

# Target output directories for split datasets
OUTPUT_DIR = "data"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VALID_DIR = os.path.join(OUTPUT_DIR, "valid")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

# Dataset Split Configuration (Sum must equal 1.0)
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# Audio Processing Configuration
TARGET_SAMPLE_RATE = 16000

# Random Seed for Reproducible Splitting
RANDOM_SEED = 42
# ==============================================================================

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "preprocess.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_md5(file_path: str) -> str:
    """
    Calculate MD5 checksum of a file to detect duplicates.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate MD5 for {file_path}: {e}")
        return ""


def scan_directory(directory: str, allowed_extensions: Set[str]) -> Dict[str, str]:
    """
    Scan directory recursively for files with matching extensions, mapping stem to absolute path.
    Specifically handles stripping the '_corrected' suffix for annotation matches.
    """
    file_map = {}
    if not os.path.exists(directory):
        logger.warning(f"Source directory does not exist: {directory}")
        return file_map

    for root, _, files in os.walk(directory):
        for file in files:
            name, ext = os.path.splitext(file)
            ext_lower = ext.lower()
            if ext_lower in allowed_extensions:
                # Strip '_corrected' from the stem map key to align annotations with audio/transcripts
                match_name = name[:-10] if name.endswith("_corrected") else name
                full_path = os.path.abspath(os.path.join(root, file))
                if match_name in file_map:
                    logger.warning(
                        f"Duplicate stem '{match_name}' encountered in {directory}. "
                        f"Overwriting {file_map[match_name]} with {full_path}"
                    )
                file_map[match_name] = full_path
    return file_map


def normalize_transcript(text: str) -> str:
    """
    Trim whitespace, collapse repeated spaces, preserve original casing, and
    retain only valid alphabetic characters, digits, spaces, and standard punctuation.
    Filters out characters that break Wav2Vec2 training pipelines.
    """
    # Trim leading/trailing spaces
    text = text.strip()
    
    # Replace hyphens, underscores and slashes with space (standard for word segmentation)
    text = text.replace("-", " ").replace("_", " ").replace("/", " ")
    
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text)
    
    # Retain only standard casing characters, digits, spaces, apostrophes, commas, periods, question marks, exclamation marks
    text = re.sub(r"[^a-zA-Z0-9\s',.?!]", "", text)
    
    # Collapse multiple whitespaces again (in case characters were stripped)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_speaker_id(stem: str, annotation_path: str) -> str:
    """
    Gracefully returns empty string since speaker IDs are not present in this dataset.
    """
    return ""


def validate_annotation_integrity(annotation_path: str, transcript_words: List[str], audio_duration: float) -> Tuple[bool, str]:
    """
    Validate word-level annotation text file.
    Checks:
    1. Check if the file is not empty.
    2. Valid structure: exactly 3 fields per line (start_time, end_time, word).
    3. Timestamps are numeric.
    4. Time ranges: start >= 0 and end_time > start_time.
    5. Causality/Chronological validation: timestamps increase monotonically.
    6. Time ranges: end_time <= audio_duration + 0.1s (floating point tolerance).
    """
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return False, f"Failed to read annotation file: {e}"

    if not lines:
        return False, "Annotation file is empty"
    annotation_words = []
    previous_end = -0.01
    for idx, line in enumerate(lines):
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            return False, f"Line {idx + 1} does not contain exactly three fields: got {len(parts)}"

        start_str, end_str, word = parts
        annotation_words.append(word)

        try:
            start = float(start_str)
            end = float(end_str)
        except ValueError:
            return False, f"Line {idx + 1} contains non-numeric timestamps: '{start_str}', '{end_str}'"

        if start < 0:
            return False, f"Negative start time on line {idx + 1}: {start}"
        if end <= start:
            return False, f"End time must be greater than start time on line {idx + 1}: start={start}, end={end}"
        if end > audio_duration + 0.1:
            return False, f"End time exceeds audio duration ({audio_duration}s) on line {idx + 1}: end={end}"
        
        # Chronological validation
        EPS = 1e-6

        if start + EPS < previous_end:
            return False, (
                f"Annotation timestamps not chronological: "
                f"line {idx + 1} starts at {start} before previous ends at {previous_end}"
            )

    if annotation_words != transcript_words:
        return False, "Transcript words do not match annotation words"

    return True, "Valid"


def verify_single_stem(stem: str, audio_path: str, transcript_path: str, anno_path: str) -> Dict:
    """
    Validates a single stem's audio file, transcript text, and plain text annotations.
    Calculates MD5 hash and determines if resampling/downmixing is needed.
    """
    try:
        # Check empty files
        if os.path.getsize(audio_path) == 0:
            return {"stem": stem, "status": "removed", "reason": "Empty audio file"}
        if os.path.getsize(transcript_path) == 0:
            return {"stem": stem, "status": "removed", "reason": "Empty transcript file"}
        if os.path.getsize(anno_path) == 0:
            return {"stem": stem, "status": "removed", "reason": "Empty annotation file"}

        # Read transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        norm_text = normalize_transcript(raw_text)
        if not norm_text:
            return {"stem": stem, "status": "removed", "reason": "Transcript has no valid characters after normalization"}

        # Read audio metadata
        waveform, sr = sf.read(audio_path, always_2d=True)
        waveform = torch.from_numpy(waveform.T).float()
        duration = waveform.shape[1] / sr
        channels = waveform.shape[0]

        if duration <= 0:
            return {"stem": stem, "status": "removed", "reason": "Audio duration is zero or negative"}

        # Validate word-level annotations (plain-text space separated format)
        is_anno_valid, anno_reason = validate_annotation_integrity(anno_path, norm_text.split(), duration)
        if not is_anno_valid:
            return {"stem": stem, "status": "removed", "reason": f"Invalid annotation: {anno_reason}"}

        # Extract speaker ID
        speaker_id = extract_speaker_id(stem, anno_path)

        # Calculate MD5 checksum
        md5_hash = calculate_md5(audio_path)

        # Determine if resampling or format changes are required
        ext = os.path.splitext(audio_path)[1].lower()
        needs_conversion = not (ext == ".wav" and sr == TARGET_SAMPLE_RATE and channels == 1)

        return {
            "stem": stem,
            "status": "valid",
            "audio_path": audio_path,
            "transcript_path": transcript_path,
            "anno_path": anno_path,
            "transcript": norm_text,
            "duration": duration,
            "sample_rate": sr,
            "channels": channels,
            "speaker_id": speaker_id,
            "md5": md5_hash,
            "needs_conversion": needs_conversion
        }

    except Exception as e:
        return {"stem": stem, "status": "removed", "reason": f"System error during analysis: {e}"}


def process_and_save_audio(src_path: str, dest_path: str) -> bool:
    """
    Loads source audio, converts to mono, resamples to 16 kHz, 
    and saves to destination as 16-bit PCM WAV.
    """
    try:
        waveform, sr = sf.read(src_path, always_2d=True)
        waveform = torch.from_numpy(waveform.T).float()
        
        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample to 16000 Hz if necessary
        if sr != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
            waveform = resampler(waveform)
            
        # Save as 16-bit PCM WAV
        if waveform.shape[0] == 1:
            audio_array = waveform.squeeze(0).numpy()
        else:
            audio_array = waveform.T.numpy()
        sf.write(dest_path, audio_array, TARGET_SAMPLE_RATE, subtype="PCM_16")
        return True
    except Exception as e:
        logger.error(f"Error processing audio file {src_path} -> {dest_path}: {e}")
        return False


def run_preprocessing():
    logger.info("Starting dataset preprocessing pipeline...")
    for directory in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    logger.info(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz")

    # 1. Scan folders for stems
    logger.info("Scanning raw files directories...")
    audio_extensions = {".wav", ".mp3", ".flac", ".m4a"}
    transcript_extensions = {".txt"}
    annotation_extensions = {".txt"} # Swapped from .json to .txt for annotation format

    audio_files = scan_directory(RAW_AUDIO_DIR, audio_extensions)
    transcript_files = scan_directory(RAW_TRANSCRIPT_DIR, transcript_extensions)
    annotation_files = scan_directory(RAW_ANNOTATION_DIR, annotation_extensions)

    logger.info(f"Found {len(audio_files)} audio files in {RAW_AUDIO_DIR}")
    logger.info(f"Found {len(transcript_files)} transcript files in {RAW_TRANSCRIPT_DIR}")
    logger.info(f"Found {len(annotation_files)} annotation files in {RAW_ANNOTATION_DIR}")

    all_stems: Set[str] = set(audio_files.keys()) | set(transcript_files.keys()) | set(annotation_files.keys())
    logger.info(f"Total unique file stems discovered: {len(all_stems)}")

    # 2. Check for missing matches
    missing_info = []
    complete_stems = []

    for stem in all_stems:
        audio_path = audio_files.get(stem)
        transcript_path = transcript_files.get(stem)
        anno_path = annotation_files.get(stem)

        missing_types = []
        if not audio_path:
            missing_types.append("audio")
        if not transcript_path:
            missing_types.append("transcript")
        if not anno_path:
            missing_types.append("annotation")

        if missing_types:
            missing_info.append({
                "file_stem": stem,
                "missing_type": ", ".join(missing_types)
            })
        else:
            complete_stems.append((stem, audio_path, transcript_path, anno_path))

    logger.info(f"Found {len(missing_info)} stems with missing file mappings.")

    # 3. Process complete stems in parallel
    num_threads = min(os.cpu_count() * 2, 32) if os.cpu_count() else 4
    logger.info(f"Validating {len(complete_stems)} complete stems using {num_threads} parallel threads...")

    valid_raw = []
    removed_files = []  # list of dicts: file_stem, reason

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(verify_single_stem, stem, audio_path, transcript_path, anno_path): stem
            for stem, audio_path, transcript_path, anno_path in complete_stems
        }
        for idx, future in enumerate(as_completed(futures)):
            stem = futures[future]
            try:
                res = future.result()

                if res["status"] == "valid":
                    valid_raw.append(res)
                else:
                    print(f"\n❌ {stem}")
                    print(res["reason"])

                    removed_files.append({
                        "file_stem": stem,
                        "reason": res["reason"]
                    })
            except Exception as e:
                logger.error(f"Error checking stem '{stem}': {e}")
                removed_files.append({
                    "file_stem": stem,
                    "reason": f"Exception during processing: {e}"
                })

            if (idx + 1) % 5000 == 0 or (idx + 1) == len(complete_stems):
                logger.info(f"Checked integrity for {idx + 1}/{len(complete_stems)} stems...")

    # 4. Check for duplicates (Deterministic MD5 mapping)
    # Sort valid_raw by stem to resolve duplicate origin selection deterministically
    valid_raw.sort(key=lambda x: x["stem"])
    
    md5_map = {}
    duplicates = []
    final_valid_samples = []

    for res in valid_raw:
        md5 = res["md5"]
        stem = res["stem"]
        path = res["audio_path"]

        if md5 in md5_map:
            original_stem = md5_map[md5]
            original_path = ""
            for v in final_valid_samples:
                if v["stem"] == original_stem:
                    original_path = v["audio_path"]
                    break

            duplicates.append({
                "file_stem": stem,
                "file_path": path,
                "duplicate_of_path": original_path,
                "md5_hash": md5
            })
            removed_files.append({
                "file_stem": stem,
                "reason": f"Duplicate of {original_stem} (MD5 checksum collision)"
            })
        else:
            md5_map[md5] = stem
            final_valid_samples.append(res)

    total_valid = len(final_valid_samples)
    logger.info(f"Retained {total_valid} valid samples. Discarded duplicates and corrupted elements.")

    # 5. Generate Splits (Train/Validation/Test)
    random.seed(RANDOM_SEED)
    random.shuffle(final_valid_samples)

    train_end = int(TRAIN_RATIO * total_valid)
    valid_end = train_end + int(VALID_RATIO * total_valid)

    train_samples = final_valid_samples[:train_end]
    valid_samples = final_valid_samples[train_end:valid_end]
    test_samples = final_valid_samples[valid_end:]

    splits_data = [
        ("train", train_samples, TRAIN_DIR),
        ("valid", valid_samples, VALID_DIR),
        ("test", test_samples, TEST_DIR)
    ]

    logger.info("Executing dataset splits and saving metadata records...")

    for split_name, samples, split_dir in splits_data:
        split_audio_dir = os.path.join(split_dir, "audio")
        split_anno_dir = os.path.join(split_dir, "annotations")

        os.makedirs(split_audio_dir, exist_ok=True)
        os.makedirs(split_anno_dir, exist_ok=True)

        metadata_records = []

        for idx, sample in enumerate(samples):
            stem = sample["stem"]

            # Deciding audio path: Reuse raw file if it satisfies requirements, convert otherwise
            if sample["needs_conversion"]:
                dest_audio_path = os.path.abspath(os.path.join(split_audio_dir, f"{stem}.wav"))
                success = process_and_save_audio(sample["audio_path"], dest_audio_path)
                if not success:
                    logger.error(f"Failed to convert audio file for {stem}. Skipping...")
                    removed_files.append({
                        "file_stem": stem,
                        "reason": "Failed to resample/save audio file during split extraction"
                    })
                    continue
                # Save relative path from project root
                rel_audio_path = os.path.relpath(dest_audio_path, os.getcwd()).replace("\\", "/")
            else:
                dest_audio_path = os.path.abspath(
                    os.path.join(split_audio_dir, f"{stem}.wav")
                )

                shutil.copy2(sample["audio_path"], dest_audio_path)

                rel_audio_path = os.path.relpath(
                    dest_audio_path,
                    os.getcwd()
                ).replace("\\", "/")

            # Copy raw word annotation text file to split annotations directory (preserving raw filename format)
            annotation_filename = os.path.basename(sample["anno_path"])
            dest_anno_path = os.path.abspath(os.path.join(split_anno_dir, annotation_filename))
            try:
                shutil.copy2(sample["anno_path"], dest_anno_path)
                rel_anno_path = os.path.relpath(dest_anno_path, os.getcwd()).replace("\\", "/")
            except Exception as e:
                logger.error(f"Failed to copy annotation file for {stem}: {e}")
                removed_files.append({
                    "file_stem": stem,
                    "reason": f"Failed to copy annotation text file: {e}"
                })
                continue

            metadata_records.append({
                "audio_path": rel_audio_path,
                "transcript": sample["transcript"],
                "annotation_path": rel_anno_path,
                "duration": round(sample["duration"], 3),
                "sample_rate": TARGET_SAMPLE_RATE,
                "speaker_id": sample["speaker_id"],
                "split": split_name
            })

            if (idx + 1) % 5000 == 0 or (idx + 1) == len(samples):
                logger.info(f"Processed {idx + 1}/{len(samples)} samples for '{split_name}' split.")

        # Save transcripts.csv for split
        csv_path = os.path.join(split_dir, "transcripts.csv")
        df = pd.DataFrame(metadata_records)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved split metadata CSV: {csv_path} ({len(df)} entries)")

    # 6. Generate and save statistics report (dataset_statistics.json)
    total_speech_duration = sum(s["duration"] for s in final_valid_samples)
    stats = {
        "total_audio_files": len(audio_files),
        "valid_samples": len(final_valid_samples),
        "removed_samples": len(removed_files) + len(missing_info),
        "duplicate_files": len(duplicates),
        "corrupted_files": sum(
            1 for f in removed_files
            if (
                "Invalid annotation" in f["reason"]
                or "System error" in f["reason"]
                or "Empty audio" in f["reason"]
                or "Empty transcript" in f["reason"]
                or "Empty annotation" in f["reason"]
                or "Audio duration" in f["reason"]
            )
        ),
        "missing_transcripts": sum(1 for m in missing_info if "transcript" in m["missing_type"]),
        "missing_annotations": sum(1 for m in missing_info if "annotation" in m["missing_type"]),
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(valid_samples),
        "test_sample_count": len(test_samples),
        "total_speech_duration_hours": round(total_speech_duration / 3600.0, 3)
    }

    with open("dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    logger.info("Saved dataset_statistics.json successfully.")

    # 7. Generate CSV Reports
    # removed_files.csv
    all_removed = []
    for item in missing_info:
        all_removed.append({
            "file_stem": item["file_stem"],
            "reason": f"Missing file(s): {item['missing_type']}"
        })
    for item in removed_files:
        all_removed.append(item)

    df_removed = pd.DataFrame(all_removed)
    if df_removed.empty:
        df_removed = pd.DataFrame(columns=["file_stem", "reason"])
    df_removed.to_csv("removed_files.csv", index=False, encoding="utf-8")
    logger.info("Saved removed_files.csv successfully.")

    # duplicates.csv
    df_duplicates = pd.DataFrame(duplicates)
    if df_duplicates.empty:
        df_duplicates = pd.DataFrame(columns=["file_stem", "file_path", "duplicate_of_path", "md5_hash"])
    df_duplicates.to_csv("duplicates.csv", index=False, encoding="utf-8")
    logger.info("Saved duplicates.csv successfully.")

    # missing_files.csv
    df_missing = pd.DataFrame(missing_info)
    if df_missing.empty:
        df_missing = pd.DataFrame(columns=["file_stem", "missing_type"])
    df_missing.to_csv("missing_files.csv", index=False, encoding="utf-8")
    logger.info("Saved missing_files.csv successfully.")

    # 8. Print terminal summary report
    summary_title = "PREPROCESSING & REVISED DATASET SPLIT SUMMARY"
    logger.info("=" * 60)
    logger.info(f"{summary_title:^60}")
    logger.info("=" * 60)
    logger.info(f"Total Unique Stems Scanned  : {len(all_stems)}")
    logger.info(f"Valid Samples Retained      : {total_valid}")
    logger.info(f"Removed/Skipped Samples     : {len(removed_files) + len(missing_info)}")
    logger.info("-" * 60)
    logger.info(f"Missing Files Stems         : {len(missing_info)}")
    logger.info(f"Corrupted Audio Files       : {stats['corrupted_files']}")
    logger.info(f"Duplicate Content Files     : {len(duplicates)}")
    logger.info("-" * 60)
    logger.info(f"Train Split Count           : {len(train_samples)}")
    logger.info(f"Validation Split Count      : {len(valid_samples)}")
    logger.info(f"Test Split Count            : {len(test_samples)}")
    logger.info(f"Total Speech Duration       : {stats['total_speech_duration_hours']:.3f} hours")
    logger.info("=" * 60)
    logger.info("Dataset preprocessing revision completed.")


if __name__ == "__main__":
    run_preprocessing()
