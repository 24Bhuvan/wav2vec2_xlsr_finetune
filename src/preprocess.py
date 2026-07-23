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
import sys # Import sys for direct stderr printing


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

# Create handlers and set their levels explicitly
file_handler = logging.FileHandler(os.path.join("logs", "preprocess.log"), mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG) # Ensure file handler captures DEBUG messages
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG) # Ensure stream handler captures DEBUG messages

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[file_handler, stream_handler],
    force=True # Add force=True to ensure configuration is reapplied
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Explicitly set the level for the named logger


def calculate_md5(file_path: str) -> str:
    """
    Calculate MD5 checksum of a file to detect duplicates.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def _read_text_file(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning(f"Could not read {filepath}: {e}")
        return ""

def _parse_annotation_file(filepath: str) -> List[str]:
    annotations = []
    content = _read_text_file(filepath)
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 3:
            annotations.append(parts[2])
    return annotations

def _clean_text(text: str) -> str:
    # Remove text within parentheses first
    text = re.sub(r'\(.*\)', '', text)
    # Remove any character that is not a letter or space
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove apostrophes specifically, then lowercase and strip
    text = text.replace("'", "").lower().strip()
    return text

def _validate_audio_duration(audio_path: str, annotation_end_time: float) -> str:
    try:
        info = torchaudio.info(audio_path)
        audio_duration = info.num_frames / info.sample_rate
        if annotation_end_time > audio_duration:
            return f"End time exceeds audio duration ({audio_duration:.2f}s)"
    except Exception as e:
        return f"Corrupted audio file or unable to read: {e}"
    return ""

def _validate_annotation_timestamps(annotations: List[Tuple[float, float, str]]) -> str:
    prev_end = 0.0
    for i, (start, end, _) in enumerate(annotations):
        if start >= end:
            return f"End time must be greater than start time on line {i+1}: start={start}, end={end}"
        if start < prev_end:
            # Allow for very minor overlaps due to rounding, but flag significant ones
            if (prev_end - start) > 0.01: # e.g., if overlap is more than 10ms
                return f"Annotation overlap or non-sequential on line {i+1}: current start={start}, previous end={prev_end}"
        prev_end = end
    return ""

def _validate_annotation_content(file_stem: str, audio_path: str, transcript_path: str, annotation_path: str) -> Tuple[bool, str]:
    transcript_text = _read_text_file(transcript_path)
    annotation_words_list = _parse_annotation_file(annotation_path)

    if not transcript_text:
        return False, "Empty or unreadable transcript file"
    if not annotation_words_list:
        return False, "Empty or unreadable annotation file"

    # Check annotation timestamps
    raw_annotations = []
    content = _read_text_file(annotation_path)
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 3:
            try:
                raw_annotations.append((float(parts[0]), float(parts[1]), parts[2]))
            except ValueError:
                return False, "Malformed timestamp in annotation file"
    timestamp_error = _validate_annotation_timestamps(raw_annotations)
    if timestamp_error:
        return False, timestamp_error

    # Check audio duration vs. last annotation end time
    last_annotation_end_time = raw_annotations[-1][1] if raw_annotations else 0.0
    audio_duration_error = _validate_audio_duration(audio_path, last_annotation_end_time)
    if audio_duration_error:
        return False, audio_duration_error

    # Compare cleaned words
    extracted_annotation_words = " ".join(annotation_words_list)
    extracted_annotation_words_cleaned = _clean_text(extracted_annotation_words)
    cleaned_transcript_text_processed = _clean_text(transcript_text)

    if extracted_annotation_words_cleaned != cleaned_transcript_text_processed:
        # DEBUG PRINTS (now ERROR to force visibility) - Log for any mismatch
        logger.error(f"DEBUG: Mismatch for {file_stem}")
        logger.error(f"DEBUG: Annotation (repr): {repr(extracted_annotation_words_cleaned)}")
        logger.error(f"DEBUG: Transcript (repr): {repr(cleaned_transcript_text_processed)}")
        logger.error(f"DEBUG: Annotation Length: {len(extracted_annotation_words_cleaned)}")
        logger.error(f"DEBUG: Transcript Length: {len(cleaned_transcript_text_processed)}")
        for i, (char_a, char_t) in enumerate(zip(extracted_annotation_words_cleaned, cleaned_transcript_text_processed)):
            if char_a != char_t:
                # Direct print to stderr for immediate visibility in Colab output
                print(f"!!! Mismatch for {file_stem} at index {i}: Annotation char='{char_a}' ({ord(char_a)}), Transcript char='{char_t}' ({ord(char_t)})", file=sys.stderr)
                logger.error(f"DEBUG: Mismatch at index {i}: Annotation char='{char_a}' ({ord(char_a)}), Transcript char='{char_t}' ({ord(char_t)})")
                break
        else:
            if len(extracted_annotation_words_cleaned) != len(cleaned_transcript_text_processed):
                print(f"!!! Mismatch for {file_stem}: Strings are different lengths but match up to the shorter length.", file=sys.stderr)
                logger.error(f"DEBUG: Strings are different lengths but match up to the shorter length for {file_stem}")
        return False, "Transcript words do not match annotation words"

    return True, ""

def process_stem(file_stem: str) -> Dict:
    audio_path = os.path.join(RAW_AUDIO_DIR, f"{file_stem}.mp3")
    transcript_path = os.path.join(RAW_TRANSCRIPT_DIR, f"{file_stem}.txt")
    annotation_path = os.path.join(RAW_ANNOTATION_DIR, f"{file_stem}_corrected.txt")

    if not (os.path.exists(audio_path) and os.path.exists(transcript_path) and os.path.exists(annotation_path)):
        missing_type = []
        if not os.path.exists(audio_path): missing_type.append("audio")
        if not os.path.exists(transcript_path): missing_type.append("transcript")
        if not os.path.exists(annotation_path): missing_type.append("annotation")
        return {"stem": file_stem, "status": "missing", "missing_type": ", ".join(missing_type)}

    is_valid, error_message = _validate_annotation_content(file_stem, audio_path, transcript_path, annotation_path)
    if not is_valid:
        return {"stem": file_stem, "status": "invalid", "error": error_message}

    return {"stem": file_stem, "status": "valid", "audio_path": audio_path, "transcript_path": transcript_path, "annotation_path": annotation_path}

def run_preprocessing():
    logger.info("Starting dataset preprocessing pipeline...")
    logger.info(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz")

    # 1. Scan for all unique file stems
    logger.info("Scanning raw files directories...")
    audio_files = {os.path.splitext(f)[0] for f in os.listdir(RAW_AUDIO_DIR) if f.endswith('.mp3')}
    transcript_files = {os.path.splitext(f)[0] for f in os.listdir(RAW_TRANSCRIPT_DIR) if f.endswith('.txt')}
    annotation_files = {f.replace('_corrected.txt', '') for f in os.listdir(RAW_ANNOTATION_DIR) if f.endswith('_corrected.txt')}

    all_stems = sorted(list(audio_files.union(transcript_files).union(annotation_files)))

    logger.info(f"Found {len(audio_files)} audio files in {RAW_AUDIO_DIR}")
    logger.info(f"Found {len(transcript_files)} transcript files in {RAW_TRANSCRIPT_DIR}")
    logger.info(f"Found {len(annotation_files)} annotation files in {RAW_ANNOTATION_DIR}")
    logger.info(f"Total unique file stems discovered: {len(all_stems)}")

    # Identify stems that have some, but not all, corresponding files
    missing_info = []
    complete_stems = []
    for stem in all_stems:
        has_audio = stem in audio_files
        has_transcript = stem in transcript_files
        has_annotation = stem in annotation_files

        if has_audio and has_transcript and has_annotation:
            complete_stems.append(stem)
        else:
            missing_parts = []
            if not has_audio: missing_parts.append("audio")
            if not has_transcript: missing_parts.append("transcript")
            if not has_annotation: missing_parts.append("annotation")
            missing_info.append({"file_stem": stem, "missing_type": ", ".join(missing_parts)})

    logger.info(f"Found {len(missing_info)} stems with missing file mappings.")

    # 2. Validate complete stems using ThreadPoolExecutor
    logger.info(f"Validating {len(complete_stems)} complete stems using {os.cpu_count()} parallel threads...")
    valid_samples = []
    removed_files = [] # For files removed due to content/audio issues
    duplicates = []

    # Store MD5 hashes to detect duplicates
    md5_hashes = {}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_stem = {executor.submit(process_stem, stem): stem for stem in complete_stems}
        for i, future in enumerate(as_completed(future_to_stem)):
            result = future.result()
            if result["status"] == "valid":
                # Check for audio duplicates using MD5
                audio_file_path = result["audio_path"]
                current_md5 = calculate_md5(audio_file_path)
                if current_md5 in md5_hashes:
                    duplicates.append({
                        "file_stem": result["stem"],
                        "file_path": audio_file_path,
                        "duplicate_of_path": md5_hashes[current_md5],
                        "md5_hash": current_md5
                    })
                    removed_files.append({"file_stem": result["stem"], "reason": "Duplicate audio content"})
                    logger.warning(f"Duplicate audio detected for {result['stem']}. Original: {md5_hashes[current_md5]}, Duplicate: {audio_file_path}")
                else:
                    md5_hashes[current_md5] = audio_file_path
                    valid_samples.append(result)
            else:
                reason = result["error"] if "error" in result else result["status"]
                removed_files.append({"file_stem": result["stem"], "reason": reason})
                logger.error(f"Failed to validate {result['stem']}: {reason}") # Added this logging line

            if (i + 1) % 5000 == 0:
                logger.info(f"Processed {i + 1}/{len(complete_stems)} samples.")

    # Log and filter out invalid samples (includes audio issues, content mismatches etc.)
    initial_valid_count = len(valid_samples)
    logger.info(f"Initial valid samples after content validation: {initial_valid_count}")

    # 3. Prepare data for splitting
    df = pd.DataFrame(valid_samples)
    if df.empty:
        logger.error("No valid samples remaining after preprocessing. Exiting.")
        return

    # Add duration and text length for statistics
    df['duration'] = df['audio_path'].apply(lambda x: torchaudio.info(x).num_frames / torchaudio.info(x).sample_rate)
    df['text_length'] = df['transcript_path'].apply(lambda x: len(_read_text_file(x)))

    # 4. Split dataset into train, validation, and test sets
    # Ensure reproducibility
    random.seed(RANDOM_SEED)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train_size = int(TRAIN_RATIO * len(df))
    valid_size = int(VALID_RATIO * len(df))
    # test_size = len(df) - train_size - valid_size

    train_samples = df.iloc[:train_size]
    valid_samples = df.iloc[train_size:train_size + valid_size]
    test_samples = df.iloc[train_size + valid_size:]

    # 5. Save split data to new directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALID_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    def _save_split_data(split_df: pd.DataFrame, target_dir: str, split_name: str):
        logger.info(f"Saving {len(split_df)} samples for '{split_name}' split...")
        audio_target_dir = os.path.join(target_dir, "audio")
        transcript_target_dir = os.path.join(target_dir, "transcripts")
        os.makedirs(audio_target_dir, exist_ok=True)
        os.makedirs(transcript_target_dir, exist_ok=True)

        processed_count = 0
        for index, row in split_df.iterrows():
            shutil.copy(row['audio_path'], os.path.join(audio_target_dir, os.path.basename(row['audio_path'])))
            shutil.copy(row['transcript_path'], os.path.join(transcript_target_dir, os.path.basename(row['transcript_path'])))
            processed_count += 1
            if processed_count % 5000 == 0:
                logger.info(f"Processed {processed_count}/{len(split_df)} samples for '{split_name}' split.")

        # Save transcripts.csv for the split
        split_df[['stem', 'audio_path', 'transcript_path', 'duration', 'text_length']].to_csv(
            os.path.join(target_dir, "transcripts.csv"), index=False, encoding="utf-8"
        )
        logger.info(f"Saved split metadata CSV: {os.path.join(target_dir, "transcripts.csv")} ({len(split_df)} entries)")

    _save_split_data(train_samples, TRAIN_DIR, "train")
    _save_split_data(valid_samples, VALID_DIR, "valid")
    _save_split_data(test_samples, TEST_DIR, "test")

    # 6. Generate dataset statistics
    total_valid = len(df)
    stats = {
        "total_samples": total_valid,
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "test_samples": len(test_samples),
        "total_speech_duration_hours": df['duration'].sum() / 3600,
        "avg_duration_per_sample_s": df['duration'].mean(),
        "min_duration_per_sample_s": df['duration'].min(),
        "max_duration_per_sample_s": df['duration'].max(),
        "avg_text_length": df['text_length'].mean(),
        "min_text_length": df['text_length'].min(),
        "max_text_length": df['text_length'].max(),
        "corrupted_files": len([f for f in removed_files if "Corrupted audio file" in f.get('reason', '')]) # Count only audio corruption specifically
    }

    with open(os.path.join(OUTPUT_DIR, "dataset_statistics.json"), "w", encoding="utf-8") as f:
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
