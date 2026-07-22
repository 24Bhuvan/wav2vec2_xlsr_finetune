import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from validate import compute_wer_cer
from dataset import get_test_dataloader

# ==============================================================================
# CONFIGURATION - HARDCODED CONSTANTS
# ==============================================================================
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "best_model")
DEFAULT_MODEL_PATH = "pretrained/wav2vec2-xls-r-300m"
LOG_DIR = "logs"
EVALUATION_LOG_PATH = os.path.join(LOG_DIR, "evaluation.log")
REPORT_PATH = os.path.join(LOG_DIR, "final_evaluation_report.txt")
METRICS_PATH = os.path.join(LOG_DIR, "final_metrics.json")

logger = logging.getLogger("evaluation")


def _configure_logger() -> logging.Logger:
    """Configure a file-backed logger for evaluation runs."""
    if logger.handlers:
        return logger

    os.makedirs(LOG_DIR, exist_ok=True)
    handler = logging.FileHandler(EVALUATION_LOG_PATH, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_configure_logger()


def _normalize_text(text: str) -> str:
    """Normalize text for stable matching and reporting."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _detect_device() -> torch.device:
    """Return the best available computing device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_checkpoint(
    best_model_dir: str = BEST_MODEL_DIR,
    base_model_path: str = DEFAULT_MODEL_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[Optional[Wav2Vec2ForCTC], Optional[Wav2Vec2Processor], Optional[str], Optional[Dict[str, Any]]]:
    """Load the finest checkpoint from the best-model directory and restore weights."""
    _configure_logger()
    if device is None:
        device = _detect_device()

    checkpoint_path = os.path.join(best_model_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        logger.error("Best checkpoint not found at %s", checkpoint_path)
        return None, None, None, None

    logger.info("Loading processor from %s", best_model_dir)
    try:
        processor = Wav2Vec2Processor.from_pretrained(best_model_dir)
    except Exception as exc:
        logger.warning("Could not load processor from %s: %s", best_model_dir, exc)
        try:
            processor = Wav2Vec2Processor.from_pretrained(base_model_path)
        except Exception as base_exc:
            logger.error("Could not load processor from fallback path %s: %s", base_model_path, base_exc)
            return None, None, None, None

    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0

    

    logger.info("Restoring state from checkpoint %s", checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        logger.error("Failed to load checkpoint payload: %s", exc)
        return None, None, None, None

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        logger.error("Checkpoint does not contain model_state_dict.")
        return None, None, None, None

    # Instantiate the model using the same processor/vocab as during training
    logger.info("Instantiating Wav2Vec2ForCTC model for evaluation")
    try:
        model = Wav2Vec2ForCTC.from_pretrained(
            base_model_path,
            ctc_loss_reduction="mean",
            pad_token_id=pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
    except Exception as exc:
        logger.error("Failed to instantiate model from %s: %s", base_model_path, exc)
        return None, None, None, None

    model.load_state_dict(state_dict)
    model.to(device)
    logger.info("Best checkpoint loaded successfully.")
    return model, processor, checkpoint_path, checkpoint


def evaluate_model(
    model: torch.nn.Module,
    processor: Wav2Vec2Processor,
    test_loader: Any,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, Any]:
    """Evaluate the best model on the held-out test split and compute aggregate metrics."""
    _configure_logger()
    model.eval()

    if test_loader is None:
        logger.error("Test dataloader is None; evaluation could not proceed.")
        return {"status": "error", "loss": float("inf"), "wer": 1.0, "cer": 1.0, "accuracy": 0.0, "samples": 0}

    try:
        loader_length = len(test_loader)
    except Exception as exc:
        logger.warning("Could not determine test loader length: %s", exc)
        loader_length = 0

    if loader_length == 0:
        logger.warning("Test dataset is empty; skipping evaluation.")
        return {"status": "warning", "loss": float("inf"), "wer": 1.0, "cer": 1.0, "accuracy": 0.0, "samples": 0}

    start_time = time.time()
    total_loss = 0.0
    processed_batches = 0
    total_wer_distance = 0.0
    total_wer_reference_words = 0
    total_cer_distance = 0.0
    total_cer_reference_chars = 0
    total_correct = 0
    total_samples = 0
    skipped_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
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

                if outputs.loss is not None:
                    total_loss += float(outputs.loss.detach().item())
                    processed_batches += 1

                logits = outputs.logits
                predicted_ids = torch.argmax(logits, dim=-1)
                decoded_predictions = processor.batch_decode(predicted_ids, output_char_offsets=False)
                transcripts = batch.get("transcripts", [])

                for reference, hypothesis in zip(transcripts, decoded_predictions):
                    if not isinstance(reference, str) or not isinstance(hypothesis, str):
                        logger.warning("Skipping invalid evaluation sample with non-string values.")
                        continue

                    reference_text = _normalize_text(reference)
                    hypothesis_text = _normalize_text(hypothesis)

                    reference_words = reference_text.split()
                    total_wer_reference_words += max(1, len(reference_words))
                    total_cer_reference_chars += max(1, len(reference_text))

                    try:
                        wer, cer = compute_wer_cer(reference, hypothesis)
                        total_wer_distance += wer * max(1, len(reference_words))
                        total_cer_distance += cer * max(1, len(reference_text))
                    except Exception as exc:
                        logger.warning("WER/CER computation failed for one sample: %s", exc)
                        continue

                    total_samples += 1
                    total_correct += int(reference_text == hypothesis_text)
            except Exception as exc:
                skipped_batches += 1
                logger.warning("Skipping evaluation batch %d due to error: %s", batch_idx, exc)
                continue

    avg_loss = total_loss / max(1, processed_batches)
    wer = total_wer_distance / max(1, total_wer_reference_words)
    cer = total_cer_distance / max(1, total_cer_reference_chars)
    accuracy = total_correct / max(1, total_samples)
    evaluation_time = time.time() - start_time

    result = {
        "status": "ok",
        "loss": avg_loss,
        "wer": wer,
        "cer": cer,
        "accuracy": accuracy,
        "samples": total_samples,
        "processed_batches": processed_batches,
        "skipped_batches": skipped_batches,
        "evaluation_time_seconds": evaluation_time,
    }
    logger.info(
        "Evaluation complete | Samples: %d | Loss: %.4f | WER: %.4f | CER: %.4f | Accuracy: %.4f | Time: %.2fs",
        result["samples"],
        result["loss"],
        result["wer"],
        result["cer"],
        result["accuracy"],
        result["evaluation_time_seconds"],
    )
    return result


def write_evaluation_report(metrics: Dict[str, Any], checkpoint_path: Optional[str], model_path: str = DEFAULT_MODEL_PATH) -> None:
    """Save a human-readable evaluation report and JSON metrics summary to the logs directory."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "Final Evaluation Report",
        "=======================",
        f"Timestamp: {timestamp}",
        f"Model checkpoint: {checkpoint_path or 'N/A'}",
        f"Base model path: {model_path}",
        f"Test samples: {metrics.get('samples', 0)}",
        f"Average test loss: {metrics.get('loss', float('inf')):.6f}",
        f"WER: {metrics.get('wer', 1.0):.6f}",
        f"CER: {metrics.get('cer', 1.0):.6f}",
        f"Accuracy: {metrics.get('accuracy', 0.0):.6f}",
        f"Evaluation time (s): {metrics.get('evaluation_time_seconds', 0.0):.6f}",
        f"Processed batches: {metrics.get('processed_batches', 0)}",
        f"Skipped batches: {metrics.get('skipped_batches', 0)}",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Evaluation report written to %s", REPORT_PATH)
    logger.info("Evaluation metrics written to %s", METRICS_PATH)


def run_evaluation() -> Dict[str, Any]:
    """Run the full evaluation pipeline end to end."""
    _configure_logger()
    logger.info("Starting final evaluation pipeline.")
    device = _detect_device()
    logger.info("Using device: %s", device)

    model, processor, checkpoint_path, _ = load_best_checkpoint(device=device)
    if model is None or processor is None:
        logger.error("Evaluation aborted because the best checkpoint could not be loaded.")
        return {"status": "error", "loss": float("inf"), "wer": 1.0, "cer": 1.0, "accuracy": 0.0, "samples": 0}

    test_loader = get_test_dataloader(
        processor=processor,
        batch_size=4,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        project_root=os.getcwd(),
    )
    metrics = evaluate_model(model=model, processor=processor, test_loader=test_loader, device=device, use_amp=False)
    write_evaluation_report(metrics=metrics, checkpoint_path=checkpoint_path)
    return metrics


def main() -> None:
    """Entry point for manual evaluation execution."""
    run_evaluation()


if __name__ == "__main__":
    main()
