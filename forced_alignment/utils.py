# forced_alignment/utils.py
import os
import re
import json
import csv
import logging
from datetime import datetime
import torch
import torchaudio

# ==========================================
# 1. System & Device Configuration Helpers
# ==========================================

def get_optimal_device():
    """Identifies and returns the most performant compute backend hardware available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Support for Apple Silicon acceleration
    return torch.device("cpu")

# ==========================================
# 2. Filesystem Path Validation Helpers
# ==========================================

def validate_path(path, is_directory=False, create_if_missing=False):
    """
    Verifies target path access permissions. 
    Optionally instantiates directories on the fly.
    """
    if create_if_missing and is_directory:
        os.makedirs(path, exist_ok=True)
        return True
        
    if not os.path.exists(path):
        return False
        
    if is_directory and not os.path.isdir(path):
        return False
    if not is_directory and not os.path.isfile(path):
        return False
        
    return True

# ==========================================
# 3. Audio Extraction & Standardization
# ==========================================

def load_and_standardize_audio(audio_path, target_sr=16000):
    """
    Loads an audio source, forces a single-channel downmix, 
    and handles dynamic resampling down to standard pipeline rates.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Target clip array missing at: {audio_path}")
        
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Force single-channel downmix
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Execute structural resample transformations
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        
    return waveform, target_sr

def get_audio_duration(audio_path):
    """Retrieves metadata duration properties of a file without parsing array blocks."""
    info = torchaudio.info(audio_path)
    return info.num_frames / info.sample_rate

# ==========================================
# 4. Text Normalization Helpers
# ==========================================

def normalize_transcript(text):
    """
    Cleans structural text anomalies to align standard transcripts 
    cleanly with the CTC token character map vocabulary.
    """
    if not text:
        return ""
    # Standardize character capitalization case limits
    text = text.strip().upper()
    # Strip basic punctuation expressions leaving core word literals
    text = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?\¿¡\[\]\"\'\\]", "", text)
    # Collapse multi-space gaps into uniform single space frames
    text = re.sub(r"\s+", " ", text)
    return text

# ==========================================
# 5. Math Time-to-Frame Scale Mappers
# ==========================================

def frame_to_seconds(frame_index, frame_duration):
    """Calculates the temporal beginning location point of an index."""
    return round(frame_index * frame_duration, 3)

def seconds_to_frame(seconds, frame_duration):
    """Calculates the absolute closest frame offset step index matching a moment."""
    return int(round(seconds / frame_duration))

# ==========================================
# 6. JSON Data Persistence Interfaces
# ==========================================

def save_json(data, target_file_path, indent=4):
    """Writes objects cleanly into isolated JSON blocks on disk."""
    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
    with open(target_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def load_json(target_file_path):
    """Parses local serialization files into active Python dictionaries."""
    if not os.path.exists(target_file_path):
        return None
    with open(target_file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ==========================================
# 7. CSV Serialization Records Handlers
# ==========================================

def write_csv_rows(target_file_path, headers, rows_data):
    """Commits flat metric matrices out to unified analytical spreadsheets."""
    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
    with open(target_file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(rows_data)

# ==========================================
# 8. Decoupled Diagnostic Logging Setup
# ==========================================

def setup_pipeline_logger(log_dir="./outputs/logs/"):
    """Spins up decoupled logger processes mapping standard logs out to persistent tracking files."""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Instantiate clean logger contexts without polluting third-party libraries
    logger = logging.getLogger("alignment_pipeline")
    logger.setLevel(logging.INFO)
    
    # Clear preexisting legacy streaming references
    if logger.hasHandlers():
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d]: %(message)s")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    return logger
