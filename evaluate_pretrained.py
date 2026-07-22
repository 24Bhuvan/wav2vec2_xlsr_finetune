import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import os
import json
import logging
import re
import time
from datetime import datetime

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from dataset import get_test_dataloader
from validate import compute_wer_cer


# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_PATH = "pretrained/wav2vec2-xls-r-300m"

LOG_DIR = "logs"

REPORT_PATH = os.path.join(LOG_DIR, "pretrained_evaluation_report.txt")
METRICS_PATH = os.path.join(LOG_DIR, "pretrained_metrics.json")
LOG_PATH = os.path.join(LOG_DIR, "pretrained_evaluation.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained():

    logger.info("Loading pretrained model...")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)

    model.to(device)

    model.eval()

    return model, processor


def evaluate(model, processor):

    loader = get_test_dataloader(
        processor=processor,
        batch_size=4,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        project_root=os.getcwd(),
    )

    total_loss = 0

    total_batches = 0

    total_correct = 0

    total_samples = 0

    total_wer = 0

    total_cer = 0

    start = time.time()

    with torch.inference_mode():

        for batch in loader:

            input_values = batch["input_values"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            total_batches += 1

            pred_ids = torch.argmax(outputs.logits, dim=-1)

            preds = processor.batch_decode(
                pred_ids,
                skip_special_tokens=True,
            )

            refs = batch["transcripts"]

            for ref, pred in zip(refs, preds):

                wer, cer = compute_wer_cer(ref, pred)

                total_wer += wer

                total_cer += cer

                if normalize(ref) == normalize(pred):
                    total_correct += 1

                total_samples += 1

    elapsed = time.time() - start

    metrics = {

        "loss": total_loss / total_batches,

        "wer": total_wer / total_samples,

        "cer": total_cer / total_samples,

        "accuracy": total_correct / total_samples,

        "samples": total_samples,

        "evaluation_time": elapsed

    }

    return metrics


def save(metrics):

    report = f"""
PRETRAINED MODEL EVALUATION
===========================

Timestamp : {datetime.now()}

Model : {MODEL_PATH}

Samples : {metrics['samples']}

Loss : {metrics['loss']:.4f}

WER : {metrics['wer']:.4f}

CER : {metrics['cer']:.4f}

Accuracy : {metrics['accuracy']:.4f}

Evaluation Time : {metrics['evaluation_time']:.2f} sec
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(report)


def main():

    model, processor = load_pretrained()

    metrics = evaluate(model, processor)

    save(metrics)


if __name__ == "__main__":
    main()
