import argparse
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, get_scheduler

from dataset import get_train_dataloader, get_valid_dataloader
from processor_builder import build_or_load_processor
from validate import run_validation

# ==============================================================================
# CONFIGURATION - HARDCODED CONSTANTS
# ==============================================================================
SEED = 42
DEFAULT_MODEL_PATH = "pretrained/wav2vec2-xls-r-300m"

# Optimizer Constants
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
ADAM_EPSILON = 1e-8
ADAM_BETAS = (0.9, 0.999)

# Scheduler Constants
WARMUP_STEPS = None
TOTAL_TRAINING_STEPS = 10000

# Training Constants
BATCH_SIZE = 8
MAX_EPOCHS = 50
GRADIENT_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 10
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_DIR = os.path.join(CHECKPOINT_DIR, "best_model")
LOG_DIR = "logs"

# Setup structured logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "train.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("train")


def set_seed(seed: int = SEED) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    logger.info(f"Initializing random seeds to {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seeds set successfully.")


def setup_device() -> Tuple[torch.device, Dict[str, Any]]:
    """Detect the available hardware device and expose a concise device summary."""
    logger.info("Detecting hardware device capabilities...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        properties = torch.cuda.get_device_properties(0)
        total_memory_gb = properties.total_memory / (1024**3)
        device_info = {
            "type": "cuda",
            "name": device_name,
            "cuda_version": cuda_version,
            "total_memory_gb": round(total_memory_gb, 2),
        }
        logger.info(f"CUDA GPU detected: {device_name}")
        logger.info(f"CUDA Version: {cuda_version}")
        logger.info(f"Total GPU Memory: {total_memory_gb:.2f} GB")
    else:
        device = torch.device("cpu")
        device_info = {
            "type": "cpu",
            "name": "CPU Fallback",
            "cuda_version": "N/A",
            "total_memory_gb": "N/A",
        }
        logger.warning("CUDA GPU is not available. Falling back to CPU training.")
    return device, device_info


def load_model_and_processor(
    model_path_or_name: str = DEFAULT_MODEL_PATH,
) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor, Dict[str, Any]]:
    """Load the model and processor, and freeze the feature encoder for fine-tuning."""
    logger.info(f"Loading Wav2Vec2Processor from: {model_path_or_name}...")
    # Build vocabulary from training transcripts on first run;
    # reload the saved processor on every subsequent run. No HF downloads.
    processor = build_or_load_processor(
        processor_dir=model_path_or_name,
        train_csv_path="data/train/transcripts.csv",
    )

    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0

    logger.info(f"Loading Wav2Vec2ForCTC from: {model_path_or_name}...")
    try:
        model = Wav2Vec2ForCTC.from_pretrained(
            model_path_or_name,
            ctc_loss_reduction="mean",
            pad_token_id=pad_token_id,
            vocab_size=len(processor.tokenizer),
        )
    except Exception as exc:
        logger.warning(
            f"Could not load model from local path '{model_path_or_name}' ({exc}). "
            "Falling back to Hugging Face Hub."
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",
            ctc_loss_reduction="mean",
            pad_token_id=pad_token_id,
            vocab_size=len(processor.tokenizer),
        )

    logger.info("Freezing CNN feature encoder layers...")
    model.freeze_feature_encoder()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    param_summary = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
    }

    logger.info(f"Model successfully loaded. Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} (Frozen: {frozen_params:,})")
    return model, processor, param_summary


def initialize_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    """Initialize AdamW with decoupled weight decay for trainable parameters."""
    logger.info("Initializing AdamW optimizer...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has no trainable parameters. Optimizer cannot be initialized.")

    decay_params: List[torch.nn.Parameter] = []
    nodecay_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in ["bias", "LayerNorm.weight"]):
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
        eps=ADAM_EPSILON,
        betas=ADAM_BETAS,
    )
    logger.info(f"Optimizer configured with learning rate: {LEARNING_RATE}, weight decay: {WEIGHT_DECAY}")
    return optimizer


def initialize_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int = TOTAL_TRAINING_STEPS,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Initialize a linear warmup scheduler for the optimizer."""
    logger.info("Initializing linear learning rate scheduler...")
    logger.info(f"Warmup steps: {warmup_steps}, total training steps: {total_steps}")
    return get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def check_mixed_precision() -> Tuple[bool, str]:
    """Return whether AMP can be enabled and a human-readable status string."""
    logger.info("Checking mixed-precision training capabilities...")
    if not torch.cuda.is_available():
        return False, "Not available (CPU fallback)"

    bf16_supported = torch.cuda.is_bf16_supported()
    status = f"Available (FP16: Supported, BF16: {'Supported' if bf16_supported else 'Not Supported'})"
    logger.info(f"Mixed precision status: {status}")
    return True, status


def print_startup_summary(device_info: Dict[str, Any], param_summary: Dict[str, Any], mixed_precision: str) -> None:
    """Print a concise startup summary for the training run."""
    title = "TRAINING INITIALIZATION SUMMARY"
    logger.info("=" * 60)
    logger.info(f"{title:^60}")
    logger.info("=" * 60)
    logger.info(f"Model Name            : facebook/wav2vec2-xls-r-300m")
    logger.info(f"Loaded Path           : {DEFAULT_MODEL_PATH}")
    logger.info(f"Device                : {device_info['type'].upper()} ({device_info['name']})")
    if device_info["type"] == "cuda":
        logger.info(f"  CUDA Version        : {device_info['cuda_version']}")
        logger.info(f"  GPU VRAM            : {device_info['total_memory_gb']} GB")
    logger.info("-" * 60)
    logger.info(f"Total Parameters     : {param_summary['total_parameters']:,}")
    logger.info(f"Trainable Parameters : {param_summary['trainable_parameters']:,}")
    logger.info(f"Frozen Parameters    : {param_summary['frozen_parameters']:,}")
    logger.info("-" * 60)
    logger.info(f"Batch Size (Config)  : {BATCH_SIZE}")
    logger.info(f"Learning Rate        : {LEARNING_RATE}")
    logger.info(f"Weight Decay         : {WEIGHT_DECAY}")
    logger.info(f"Optimizer            : AdamW (eps={ADAM_EPSILON}, betas={ADAM_BETAS})")
    logger.info(f"Scheduler            : Linear Warmup (10% total steps)")
    logger.info(f"Total Train Steps    : {TOTAL_TRAINING_STEPS}")
    logger.info("-" * 60)
    logger.info(f"Mixed Precision      : {mixed_precision}")
    logger.info("=" * 60)


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    train_loss: float,
    val_loss: float,
    wer: float,
    cer: float,
    learning_rate: float,
    best_val_loss: float,
    best_epoch: int,
    history: List[Dict[str, Any]],
) -> None:
    """Persist a checkpoint containing model, optimizer, scheduler, and metrics."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "wer": wer,
        "cer": cer,
        "learning_rate": learning_rate,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
    }
    torch.save(payload, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> Tuple[int, float, int, List[Dict[str, Any]]]:
    """Restore training state from an existing checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    history = checkpoint.get("history", [])
    logger.info(f"Resuming from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
    return start_epoch, best_val_loss, best_epoch, history


def append_history(history_path: str, entry: Dict[str, Any]) -> None:
    """Persist a training history entry as JSONL for downstream inspection."""
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def train_model(
    model: torch.nn.Module,
    processor: Wav2Vec2Processor,
    device: torch.device,
    max_epochs: int = MAX_EPOCHS,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Run the full fine-tuning loop with validation, checkpointing, and early stopping."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    history_path = os.path.join(LOG_DIR, "training_history.jsonl")

    # Bug fix: Only remove history log if starting fresh (not resuming)
    if not checkpoint_path and os.path.exists(history_path):
        os.remove(history_path)

    train_loader = get_train_dataloader(
        processor=processor,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        project_root=os.getcwd(),
    )
    if len(train_loader) == 0:
        raise ValueError("Training DataLoader is empty. Verify that data/train/transcripts.csv exists and contains valid entries.")
    valid_loader = get_valid_dataloader(
        processor=processor,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        project_root=os.getcwd(),
    )

    optimizer = initialize_optimizer(model)
    num_training_steps = max(1, max_epochs * len(train_loader))
    
    # Dynamically compute warmup steps (10% of total training steps)
    warmup_steps = int(0.1 * num_training_steps)
    scheduler = initialize_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=num_training_steps)

    use_amp, mixed_precision = check_mixed_precision()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    history: List[Dict[str, Any]] = []

    if checkpoint_path:
        start_epoch, best_val_loss, best_epoch, history = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    logger.info("Training pipeline initialized. Starting fine-tuning...")
    total_start_time = time.time()
    epochs_without_improvement = 0

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader, start=1):
            input_values = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                optimizer.step()

            scheduler.step()
            batch_loss = loss.detach().item()
            epoch_total_loss += batch_loss

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d | Batch %d/%d | Loss: %.4f | LR: %.6f",
                epoch,
                max_epochs,
                batch_idx,
                len(train_loader),
                batch_loss,
                current_lr,
            )

        avg_train_loss = epoch_total_loss / max(1, len(train_loader))
        val_metrics = run_validation(
            model=model,
            val_loader=valid_loader,
            processor=processor,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            best_val_loss=best_val_loss,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        if val_metrics["loss"] < best_val_loss - 1e-6:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_checkpoint_path = os.path.join(BEST_MODEL_DIR, "checkpoint.pt")
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_metrics["loss"],
                wer=val_metrics["wer"],
                cer=val_metrics["cer"],
                learning_rate=optimizer.param_groups[0]["lr"],
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                history=history,
            )
            
            # Export standard Hugging Face model and processor format
            model.save_pretrained(BEST_MODEL_DIR)
            processor.save_pretrained(BEST_MODEL_DIR)
            
            logger.info(f"New best validation loss: {best_val_loss:.4f} at epoch {epoch}")
        else:
            epochs_without_improvement += 1

        epoch_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:02d}", "checkpoint.pt")
        save_checkpoint(
            checkpoint_path=epoch_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_metrics["loss"],
            wer=val_metrics["wer"],
            cer=val_metrics["cer"],
            learning_rate=optimizer.param_groups[0]["lr"],
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            history=history,
        )

        epoch_time = time.time() - epoch_start_time
        history_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "validation_loss": round(val_metrics["loss"], 6),
            "wer": round(val_metrics["wer"], 6),
            "cer": round(val_metrics["cer"], 6),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
            "time_per_epoch_seconds": round(epoch_time, 2),
        }
        history.append(history_entry)
        append_history(history_path, history_entry)

        logger.info(
            "Epoch %d/%d complete | Train Loss: %.4f | Val Loss: %.4f | WER: %.4f | CER: %.4f | LR: %.6f | Time: %.2fs",
            epoch,
            max_epochs,
            avg_train_loss,
            val_metrics["loss"],
            val_metrics["wer"],
            val_metrics["cer"],
            optimizer.param_groups[0]["lr"],
            epoch_time,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logger.info(
                "Early stopping triggered because validation loss has not improved for %d epochs.",
                EARLY_STOPPING_PATIENCE,
            )
            break

    total_training_time = time.time() - total_start_time
    logger.info("Training completed.")
    logger.info(f"Total epochs completed: {len(history)}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Lowest validation loss: {best_val_loss:.4f}")
    logger.info(f"Final WER: {history[-1]['wer']:.4f}")
    logger.info(f"Final CER: {history[-1]['cer']:.4f}")
    logger.info(f"Total training time: {total_training_time:.2f} seconds")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2-XLS-R 300M")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Maximum number of epochs to train")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to a checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    """Initialize the training environment and execute the training loop."""
    args = parse_args()

    # Automatic checkpoint discovery if not provided
    if args.checkpoint_path is None:
        if os.path.exists(CHECKPOINT_DIR):
            checkpoint_dirs = [
                d for d in os.listdir(CHECKPOINT_DIR)
                if os.path.isdir(os.path.join(CHECKPOINT_DIR, d)) and d.startswith("epoch_")
            ]
            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda x: int(x.split("_")[1]))
                latest_epoch_dir = checkpoint_dirs[-1]
                args.checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_epoch_dir, "checkpoint.pt")
                logger.info(f"Latest checkpoint detected: {args.checkpoint_path}")
                logger.info(f"Resuming training from epoch {int(latest_epoch_dir.split('_')[1]) + 1}.")
            else:
                logger.info("No checkpoint found. Starting training from epoch 1.")
        else:
            logger.info("No checkpoint directory found. Starting training from epoch 1.")
    else:
        logger.info(f"Resuming from user-specified checkpoint: {args.checkpoint_path}")

    set_seed(SEED)
    device, device_info = setup_device()
    model, processor, param_summary = load_model_and_processor(DEFAULT_MODEL_PATH)
    model.to(device)
    mixed_precision = check_mixed_precision()[1]
    print_startup_summary(device_info, param_summary, mixed_precision)
    train_model(model=model, processor=processor, device=device, max_epochs=args.epochs, checkpoint_path=args.checkpoint_path)


if __name__ == "__main__":
    main()
