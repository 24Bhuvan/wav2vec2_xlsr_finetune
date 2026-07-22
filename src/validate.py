import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import Wav2Vec2Processor

# ==============================================================================
# CONFIGURATION - HARDCODED CONSTANTS
# ==============================================================================
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "best_model")
LOG_DIR = "logs"
VALIDATION_LOG_PATH = os.path.join(LOG_DIR, "validation_results.jsonl")

logger = logging.getLogger("validation")


def _configure_logger() -> logging.Logger:
    """Configure a file-backed validation logger once per process."""
    if logger.handlers:
        return logger

    os.makedirs(LOG_DIR, exist_ok=True)
    handler = logging.FileHandler(os.path.join(LOG_DIR, "validation.log"), mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_configure_logger()


def _normalize_text(text: str) -> str:
    """Normalize transcripts for stable WER/CER comparisons."""
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous_row = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current_row = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (0 if ref_char == hyp_char else 1)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def compute_wer_cer(reference: str, hypothesis: str) -> Tuple[float, float]:
    """Compute word error rate and character error rate for a single text pair."""
    try:
        import jiwer

        normalized_reference = _normalize_text(reference)
        normalized_hypothesis = _normalize_text(hypothesis)
        wer = float(jiwer.wer(normalized_reference, normalized_hypothesis))
        cer = float(jiwer.cer(normalized_reference, normalized_hypothesis))
        return wer, cer
    except Exception:
        ref_words = _normalize_text(reference).split()
        hyp_words = _normalize_text(hypothesis).split()
        if len(ref_words) == 0 and len(hyp_words) == 0:
            wer = 0.0
        else:
            wer = _levenshtein_distance(" ".join(ref_words), " ".join(hyp_words)) / max(1, len(ref_words))

        ref_chars = _normalize_text(reference)
        hyp_chars = _normalize_text(hypothesis)
        if len(ref_chars) == 0 and len(hyp_chars) == 0:
            cer = 0.0
        else:
            cer = _levenshtein_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))
        return wer, cer


def _append_validation_log(entry: Dict[str, Any], log_path: str = VALIDATION_LOG_PATH) -> None:
    """Append validation metrics as JSONL for downstream analysis."""
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def save_best_model(
    model: torch.nn.Module,
    processor: Wav2Vec2Processor,
    best_model_dir: str = BEST_MODEL_DIR,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> str:
    """Persist the best validation checkpoint, processor, and tokenizer to disk."""
    os.makedirs(best_model_dir, exist_ok=True)

    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    checkpoint_path = os.path.join(best_model_dir, "checkpoint.pt")
    torch.save(payload, checkpoint_path)

    try:
        processor.save_pretrained(best_model_dir)
    except Exception as exc:
        logger.warning("Could not save processor to %s: %s", best_model_dir, exc)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(os.path.join(best_model_dir, "tokenizer"))
        except Exception as exc:
            logger.warning("Could not save tokenizer to %s: %s", os.path.join(best_model_dir, "tokenizer"), exc)

    logger.info("Best model checkpoint saved to %s", best_model_dir)
    return checkpoint_path


def run_validation(
    model: torch.nn.Module,
    val_loader: Any,
    processor: Wav2Vec2Processor,
    device: torch.device,
    use_amp: bool = False,
    epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run validation over the validation split and report loss, WER, CER, and best-model status."""
    _configure_logger()
    model.eval()

    start_time = time.time()
    total_loss = 0.0
    total_wer = 0.0
    total_cer = 0.0
    processed_batches = 0
    sample_count = 0
    skipped_batches = 0
    best_model_updated = False

    if val_loader is None:
        logger.error("Validation dataloader is None; validation could not be performed.")
        result = {"loss": float("inf"), "wer": 1.0, "cer": 1.0}
        model.train()
        return result

    try:
        loader_length = len(val_loader)
    except Exception as exc:
        logger.warning("Could not determine validation loader length: %s", exc)
        loader_length = 0

    if loader_length == 0:
        logger.warning("Validation dataset is empty; skipping validation.")
        result = {"loss": float("inf"), "wer": 1.0, "cer": 1.0}
        model.train()
        return result

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader, start=1):
            try:
                input_values = batch["input_values"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                if use_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    ):
                        outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)

                if outputs.loss is None:
                    raise ValueError("Validation loss was None.")

                total_loss += float(outputs.loss.detach().item())
                processed_batches += 1

                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                decoded_predictions = processor.batch_decode(predicted_ids, output_char_offsets=False)

                transcripts = batch.get("transcripts", [])
                for reference, hypothesis in zip(transcripts, decoded_predictions):
                    if not isinstance(reference, str) or not isinstance(hypothesis, str):
                        logger.warning("Skipping invalid validation sample with non-string values.")
                        continue
                    try:
                        wer, cer = compute_wer_cer(reference, hypothesis)
                        total_wer += wer
                        total_cer += cer
                        sample_count += 1
                    except Exception as exc:
                        logger.warning("Metric computation failed for sample: %s", exc)
                        continue
            except Exception as exc:
                skipped_batches += 1
                logger.warning("Skipping validation batch %d due to error: %s", batch_idx, exc)
                continue

    avg_loss = total_loss / max(1, processed_batches)
    avg_wer = total_wer / max(1, sample_count)
    avg_cer = total_cer / max(1, sample_count)
    validation_duration = time.time() - start_time

    if best_val_loss is None:
        best_model_updated = True
    else:
        best_model_updated = avg_loss < best_val_loss - 1e-6

    logger.info(
        "Validation summary | Epoch: %s | Loss: %.4f | WER: %.4f | CER: %.4f | Duration: %.2fs | Improved: %s | Best Loss: %.4f",
        epoch if epoch is not None else "N/A",
        avg_loss,
        avg_wer,
        avg_cer,
        validation_duration,
        best_model_updated,
        avg_loss if best_model_updated else (best_val_loss if best_val_loss is not None else avg_loss),
    )

    entry = {
        "epoch": epoch,
        "validation_loss": round(avg_loss, 6),
        "wer": round(avg_wer, 6),
        "cer": round(avg_cer, 6),
        "validation_duration_seconds": round(validation_duration, 3),
        "best_model_updated": best_model_updated,
        "skipped_batches": skipped_batches,
    }
    _append_validation_log(entry)

    model.train()
    return {
        "loss": avg_loss,
        "wer": avg_wer,
        "cer": avg_cer,
    }
