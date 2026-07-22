import csv
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

# ==============================================================================
# LOGGING HELPERS
# ==============================================================================

def ensure_directory(path: str) -> str:
    """Create a directory if it does not already exist and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def configure_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Create or reuse a logger configured with a file handler and stream handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    ensure_directory(os.path.dirname(log_file) or ".")
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# ==============================================================================
# FILE SYSTEM HELPERS
# ==============================================================================

def file_exists(path: str) -> bool:
    """Return True when a file exists at the given path."""
    return os.path.isfile(path)


def directory_exists(path: str) -> bool:
    """Return True when a directory exists at the given path."""
    return os.path.isdir(path)


def read_json(path: str) -> Dict[str, Any]:
    """Read a JSON file and return its parsed contents."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a dictionary to a JSON file."""
    ensure_directory(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_csv(path: str) -> List[Dict[str, Any]]:
    """Read a CSV file and return rows as dictionaries."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write rows to a CSV file."""
    ensure_directory(os.path.dirname(path) or ".")
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ==============================================================================
# RANDOM SEED HELPERS
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# AUDIO HELPERS
# ==============================================================================

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".webm"}
TARGET_SAMPLE_RATE = 16000


def validate_audio_path(audio_path: str) -> bool:
    """Return True when the path points to an existing readable audio file."""
    if not audio_path or not file_exists(audio_path):
        return False
    extension = os.path.splitext(audio_path)[1].lower()
    return extension in SUPPORTED_AUDIO_EXTENSIONS


def load_audio_file(audio_path: str, target_sample_rate: int = TARGET_SAMPLE_RATE) -> Tuple[torch.Tensor, int]:
    """Load an audio file, convert to mono if needed, and resample to the target rate."""
    if not validate_audio_path(audio_path):
        raise FileNotFoundError(f"Audio file not found or unsupported: {audio_path}")

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.dim() > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    return waveform.squeeze(0), sample_rate


def get_audio_duration(audio_path: str) -> float:
    """Return the duration of an audio file in seconds."""
    waveform, _ = load_audio_file(audio_path)
    return waveform.shape[0] / TARGET_SAMPLE_RATE


def get_sample_rate(audio_path: str) -> int:
    """Return the sample rate of an audio file."""
    if not validate_audio_path(audio_path):
        raise FileNotFoundError(f"Audio file not found or unsupported: {audio_path}")
    _, sample_rate = torchaudio.load(audio_path)
    return sample_rate


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert a multi-channel waveform to mono by averaging channels."""
    if waveform.dim() > 1:
        return torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Resample a waveform tensor to the target sample rate."""
    if source_rate == target_rate:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=source_rate, new_freq=target_rate)
    return resampler(waveform)


# ==============================================================================
# TRANSCRIPT HELPERS
# ==============================================================================

def clean_transcript(text: str) -> str:
    """Normalize whitespace in a transcript string."""
    return re.sub(r"\s+", " ", text).strip()


def validate_transcript(text: Optional[str]) -> bool:
    """Return True when the transcript is a non-empty string after cleanup."""
    if not isinstance(text, str):
        return False
    return len(clean_transcript(text)) > 0


def is_empty_transcript(text: Optional[str]) -> bool:
    """Return True when the transcript is empty or whitespace-only."""
    return not validate_transcript(text)


# ==============================================================================
# ANNOTATION HELPERS
# ==============================================================================

def load_json_file(path: str) -> Dict[str, Any]:
    """Load and parse a JSON file as a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_annotation(annotation: Any, audio_duration: Optional[float] = None) -> bool:
    """Validate a loaded annotation payload for basic schema consistency."""
    if not isinstance(annotation, dict):
        return False
    if "segments" not in annotation and "words" not in annotation:
        return False
    if audio_duration is not None:
        for item in annotation.get("segments", []) + annotation.get("words", []):
            if isinstance(item, dict):
                start = item.get("start")
                end = item.get("end")
                if start is not None and end is not None and end > audio_duration:
                    return False
    return True


def validate_timestamps(annotation: Any, audio_duration: Optional[float] = None) -> bool:
    """Validate that annotation timestamps are non-negative and do not exceed the audio duration."""
    if not isinstance(annotation, dict):
        return False
    for item in annotation.get("segments", []) + annotation.get("words", []):
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        if start is not None and end is not None and start < 0:
            return False
        if end is not None and end < 0:
            return False
        if audio_duration is not None and end is not None and end > audio_duration:
            return False
    return True


# ==============================================================================
# CHECKPOINT HELPERS
# ==============================================================================

def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    """Save a checkpoint payload to disk."""
    ensure_directory(os.path.dirname(path) or ".")
    torch.save(payload, path)


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load a checkpoint payload from disk."""
    if not file_exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def save_best_model(path: str, model: torch.nn.Module, optimizer: Optional[Any] = None, scheduler: Optional[Any] = None, **metadata: Any) -> None:
    """Persist the model state together with optimizer, scheduler, and metadata."""
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict(), **metadata}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    save_checkpoint(path, payload)


# ==============================================================================
# METRIC HELPERS
# ==============================================================================

def average_loss(loss_values: List[float]) -> float:
    """Compute the average loss from a list of values."""
    return float(sum(loss_values) / max(1, len(loss_values))) if loss_values else 0.0


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute a simple word error rate between two strings."""
    ref_words = clean_transcript(reference).split()
    hyp_words = clean_transcript(hypothesis).split()
    if not ref_words and not hyp_words:
        return 0.0
    distance = 0
    if len(ref_words) == 0:
        return float(len(hyp_words))
    if len(hyp_words) == 0:
        return float(len(ref_words))
    previous_row = list(range(len(hyp_words) + 1))
    for i, ref_char in enumerate(ref_words, start=1):
        current_row = [i]
        for j, hyp_char in enumerate(hyp_words, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (0 if ref_char == hyp_char else 1)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    distance = previous_row[-1]
    return distance / max(1, len(ref_words))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute a simple character error rate between two strings."""
    reference_text = clean_transcript(reference).lower()
    hypothesis_text = clean_transcript(hypothesis).lower()
    if not reference_text and not hypothesis_text:
        return 0.0
    distance = 0
    previous_row = list(range(len(hypothesis_text) + 1))
    for i, ref_char in enumerate(reference_text, start=1):
        current_row = [i]
        for j, hyp_char in enumerate(hypothesis_text, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (0 if ref_char == hyp_char else 1)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    distance = previous_row[-1]
    return distance / max(1, len(reference_text))


def compute_accuracy(correct: int, total: int) -> float:
    """Compute accuracy as a ratio of correct predictions to total samples."""
    return correct / max(1, total)


# ==============================================================================
# DEVICE HELPERS
# ==============================================================================

def get_device() -> torch.device:
    """Return the active torch device, preferring CUDA when available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device) -> None:
    """Print a concise summary of the active device."""
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available; using CPU")


# ==============================================================================
# TIME HELPERS
# ==============================================================================

def format_elapsed_time(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    return time.strftime("%H:%M:%S", time.gmtime(int(seconds)))


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """Format a datetime object as a standard string."""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def estimate_remaining_time(elapsed: float, completed: int, total: int) -> str:
    """Estimate the remaining runtime based on completed and total steps."""
    if completed <= 0 or total <= 0:
        return "N/A"
    remaining = (elapsed / completed) * (total - completed)
    return format_elapsed_time(remaining)


# ==============================================================================
# GENERAL UTILITY HELPERS
# ==============================================================================

def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def format_file_size(path: str) -> str:
    """Return a human-readable file size for a file on disk."""
    if not file_exists(path):
        return "0 B"
    size_bytes = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024 or unit == "GB":
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return "0 B"


def format_progress(current: int, total: int) -> str:
    """Format progress as a percentage string."""
    if total <= 0:
        return "0%"
    return f"{(current / total) * 100:.1f}%"
