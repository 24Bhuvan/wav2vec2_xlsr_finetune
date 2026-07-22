import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ==============================================================================
# CONFIGURATION - HARDCODED CONSTANTS
# ==============================================================================
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "best_model")
INFERENCE_DIR = "inference_model"
MODEL_EXPORT_DIR = os.path.join(INFERENCE_DIR, "model")
PROCESSOR_EXPORT_DIR = os.path.join(INFERENCE_DIR, "processor")
TOKENIZER_EXPORT_DIR = os.path.join(INFERENCE_DIR, "tokenizer")
EXPORT_INFO_PATH = os.path.join(INFERENCE_DIR, "export_info.json")
DEFAULT_MODEL_PATH = "pretrained/wav2vec2-xls-r-300m"
LOG_DIR = "logs"
EXPORT_LOG_PATH = os.path.join(LOG_DIR, "export.log")

logger = logging.getLogger("export")


def _configure_logger() -> logging.Logger:
    """Configure a file-backed logger for export operations."""
    if logger.handlers:
        return logger

    os.makedirs(LOG_DIR, exist_ok=True)
    handler = logging.FileHandler(EXPORT_LOG_PATH, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_configure_logger()


def _detect_device() -> torch.device:
    """Return the best available device for model loading."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_checkpoint(
    best_model_dir: str = BEST_MODEL_DIR,
    base_model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[Wav2Vec2ForCTC], Optional[Wav2Vec2Processor], Optional[str]]:
    """Load the best checkpoint and restore the fine-tuned model weights."""
    _configure_logger()
    if device is None:
        device = _detect_device()

    checkpoint_path = os.path.join(best_model_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        logger.error("Best checkpoint not found at %s", checkpoint_path)
        return None, None, None

    try:
        processor = Wav2Vec2Processor.from_pretrained(best_model_dir)
    except Exception as exc:
        logger.warning("Could not load processor from %s: %s", best_model_dir, exc)
        try:
            processor = Wav2Vec2Processor.from_pretrained(base_model_path)
        except Exception as base_exc:
            logger.error("Could not load processor from fallback path %s: %s", base_model_path, base_exc)
            return None, None, None

    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0

    try:
        model = Wav2Vec2ForCTC.from_pretrained(
            base_model_path,
            ctc_loss_reduction="mean",
            pad_token_id=pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
    except Exception as exc:
        logger.error("Could not load base model from %s: %s", base_model_path, exc)
        return None, None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        logger.error("Unable to read checkpoint payload: %s", exc)
        return None, None, None

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        logger.error("Checkpoint does not contain model_state_dict")
        return None, None, None

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Best checkpoint loaded successfully from %s", checkpoint_path)
    return model, processor, checkpoint_path


def export_model_artifacts(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    checkpoint_path: Optional[str],
    export_dir: str = INFERENCE_DIR,
) -> Dict[str, Any]:
    """Export the model, processor, tokenizer, and export metadata to the inference directory."""
    _configure_logger()
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
    os.makedirs(PROCESSOR_EXPORT_DIR, exist_ok=True)
    os.makedirs(TOKENIZER_EXPORT_DIR, exist_ok=True)

    export_summary: Dict[str, Any] = {
        "status": "success",
        "checkpoint_path": checkpoint_path,
        "model_export_dir": MODEL_EXPORT_DIR,
        "processor_export_dir": PROCESSOR_EXPORT_DIR,
        "tokenizer_export_dir": TOKENIZER_EXPORT_DIR,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        model.save_pretrained(MODEL_EXPORT_DIR)
        export_summary["model_exported"] = True
        logger.info("Model exported to %s", MODEL_EXPORT_DIR)
    except Exception as exc:
        export_summary["model_exported"] = False
        logger.error("Model export failed: %s", exc)
        raise

    try:
        processor.save_pretrained(PROCESSOR_EXPORT_DIR)
        export_summary["processor_exported"] = True
        logger.info("Processor exported to %s", PROCESSOR_EXPORT_DIR)
    except Exception as exc:
        export_summary["processor_exported"] = False
        logger.error("Processor export failed: %s", exc)
        raise

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        logger.error("Processor does not expose a tokenizer")
        raise ValueError("Processor does not expose a tokenizer")

    try:
        tokenizer.save_pretrained(TOKENIZER_EXPORT_DIR)
        export_summary["tokenizer_exported"] = True
        logger.info("Tokenizer exported to %s", TOKENIZER_EXPORT_DIR)
    except Exception as exc:
        export_summary["tokenizer_exported"] = False
        logger.error("Tokenizer export failed: %s", exc)
        raise

    info_payload = {
        "base_model": DEFAULT_MODEL_PATH,
        "fine_tuned_model_name": "wav2vec2-xls-r-300m",
        "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "framework": {
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "export_paths": {
            "model": MODEL_EXPORT_DIR,
            "processor": PROCESSOR_EXPORT_DIR,
            "tokenizer": TOKENIZER_EXPORT_DIR,
        },
        "model_architecture": model.__class__.__name__,
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }

    with open(EXPORT_INFO_PATH, "w", encoding="utf-8") as handle:
        json.dump(info_payload, handle, indent=2)

    export_summary["export_info_path"] = EXPORT_INFO_PATH
    export_summary["export_info"] = info_payload
    logger.info("Export metadata written to %s", EXPORT_INFO_PATH)
    return export_summary


def verify_export() -> Dict[str, Any]:
    """Verify that the export directories and expected files exist."""
    _configure_logger()
    verification = {
        "model_dir_exists": os.path.isdir(MODEL_EXPORT_DIR),
        "processor_dir_exists": os.path.isdir(PROCESSOR_EXPORT_DIR),
        "tokenizer_dir_exists": os.path.isdir(TOKENIZER_EXPORT_DIR),
        "export_info_exists": os.path.isfile(EXPORT_INFO_PATH),
    }

    if verification["model_dir_exists"]:
        verification["model_files"] = os.listdir(MODEL_EXPORT_DIR)[:10]
    if verification["processor_dir_exists"]:
        verification["processor_files"] = os.listdir(PROCESSOR_EXPORT_DIR)[:10]
    if verification["tokenizer_dir_exists"]:
        verification["tokenizer_files"] = os.listdir(TOKENIZER_EXPORT_DIR)[:10]

    logger.info("Export verification complete: %s", verification)
    return verification


def save_inference_model() -> Dict[str, Any]:
    """Run the full export workflow and return a summary payload."""
    _configure_logger()
    logger.info("Starting inference model export.")
    start_time = time.time()

    device = _detect_device()
    model, processor, checkpoint_path = load_best_checkpoint(device=device)
    if model is None or processor is None:
        logger.error("Export aborted because the best checkpoint could not be loaded.")
        return {"status": "error", "message": "Best checkpoint could not be loaded."}

    try:
        export_summary = export_model_artifacts(model=model, processor=processor, checkpoint_path=checkpoint_path)
        verification = verify_export()
        duration = time.time() - start_time
        logger.info("Inference model export completed in %.2f seconds", duration)
        return {
            "status": "success",
            "duration_seconds": round(duration, 2),
            "export_summary": export_summary,
            "verification": verification,
        }
    except Exception as exc:
        logger.error("Inference model export failed: %s", exc)
        return {"status": "error", "message": str(exc)}


def main() -> None:
    """Entry point for manual export execution."""
    result = save_inference_model()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
