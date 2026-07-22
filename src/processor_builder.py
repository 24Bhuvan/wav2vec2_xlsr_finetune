"""
processor_builder.py
--------------------
Official Hugging Face CTC custom processor workflow for Wav2Vec2 fine-tuning.

Public API
----------
    build_or_load_processor(processor_dir, train_csv_path) -> Wav2Vec2Processor

Behaviour (idempotent — safe to call on every run):
  • First run  : builds vocab.json from training transcripts, creates and saves
                 Wav2Vec2CTCTokenizer + Wav2Vec2FeatureExtractor → Wav2Vec2Processor.
  • Later runs : detects saved tokenizer files and reloads the processor locally
                 without any network calls.

Only the tokenizer is custom-built.
The encoder weights (pytorch_model.bin) are loaded separately in train.py and
always come from the local pretrained directory — no Hugging Face downloads here.
"""

import json
import logging
import os
from typing import Optional

import pandas as pd
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
)

# ==============================================================================
# MODULE LOGGER
# ==============================================================================
logger = logging.getLogger("processor_builder")

# Special tokens required by Wav2Vec2 CTC
_PAD_TOKEN = "[PAD]"
_UNK_TOKEN = "[UNK]"
_WORD_DELIMITER_TOKEN = "|"

# Files that must exist for the processor to be considered fully saved
_TOKENIZER_SENTINEL_FILES = [
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


# ==============================================================================
# VOCABULARY BUILDER
# ==============================================================================

def _build_vocabulary(train_csv_path: str) -> dict:
    """
    Read every transcript from train_csv_path and build a character-level
    vocabulary suitable for Wav2Vec2 CTC training.

    Returns a {character: int_id} mapping that includes:
      - All unique characters found in the transcripts
      - The word-delimiter token '|'  (replaces spaces during CTC decoding)
      - [PAD] at index 0  (used by the CTC loss to ignore padded positions)
      - [UNK] immediately after the character set
    """
    logger.info("Creating tokenizer from transcripts: %s", train_csv_path)

    df = pd.read_csv(train_csv_path)
    if "transcript" not in df.columns:
        raise ValueError(
            f"'transcript' column not found in {train_csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Collect all unique characters across every transcript
    all_chars: set = set()
    for transcript in df["transcript"].dropna():
        text = str(transcript).strip()
        # Replace spaces with the word-delimiter token during vocab construction
        text = text.replace(" ", _WORD_DELIMITER_TOKEN)
        all_chars.update(text)

    # Remove the word-delimiter from the raw char set (we add it explicitly)
    all_chars.discard(_WORD_DELIMITER_TOKEN)

    # Sort for deterministic, reproducible vocab ordering
    sorted_chars = sorted(all_chars)

    # Assign IDs:
    #   0          → [PAD]   (CTC blank / ignore index)
    #   1 … N-1   → sorted characters
    #   N          → |       (word delimiter)
    #   N+1        → [UNK]
    vocab: dict = {_PAD_TOKEN: 0}
    for idx, char in enumerate(sorted_chars, start=1):
        vocab[char] = idx

    next_id = len(vocab)
    vocab[_WORD_DELIMITER_TOKEN] = next_id
    vocab[_UNK_TOKEN] = next_id + 1

    logger.info("Vocabulary size: %d", len(vocab))
    return vocab


# ==============================================================================
# PROCESSOR BUILDER / LOADER
# ==============================================================================

def build_or_load_processor(
    processor_dir: str,
    train_csv_path: str,
    project_root: Optional[str] = None,
) -> Wav2Vec2Processor:
    """
    Build or load a Wav2Vec2Processor for CTC fine-tuning.

    Parameters
    ----------
    processor_dir : str
        Directory that contains (or will contain) the processor files.
        Typically ``pretrained/wav2vec2-xls-r-300m``.
        Paths are resolved relative to ``project_root`` when they are
        not already absolute.
    train_csv_path : str
        Path to ``data/train/transcripts.csv``.
        Used on the first run only to build the character vocabulary.
    project_root : str, optional
        Absolute path to the repository root.  Defaults to ``os.getcwd()``.

    Returns
    -------
    Wav2Vec2Processor
        A fully initialised processor whose tokenizer vocabulary was derived
        from the training transcripts.

    Side effects (first run only)
    ------------------------------
    Writes the following files into ``processor_dir``:
        vocab.json
        tokenizer_config.json
        special_tokens_map.json
        preprocessor_config.json  (already present — overwritten with same content)
    """
    if project_root is None:
        project_root = os.getcwd()

    # Resolve paths to absolute
    if not os.path.isabs(processor_dir):
        processor_dir = os.path.abspath(os.path.join(project_root, processor_dir))
    if not os.path.isabs(train_csv_path):
        train_csv_path = os.path.abspath(os.path.join(project_root, train_csv_path))

    os.makedirs(processor_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Fast path: all tokenizer sentinel files already exist → just reload
    # ------------------------------------------------------------------
    all_saved = all(
        os.path.exists(os.path.join(processor_dir, fname))
        for fname in _TOKENIZER_SENTINEL_FILES
    )

    if all_saved:
        logger.info("Loading existing processor from: %s", processor_dir)
        processor = Wav2Vec2Processor.from_pretrained(processor_dir)
        logger.info(
            "Processor loaded. Vocabulary size: %d",
            len(processor.tokenizer),
        )
        # Sanity-check special token IDs for CTC compatibility
        logger.info(
            "pad_token_id=%s, unk_token_id=%s, word_delimiter_token=%s",
            processor.tokenizer.pad_token_id,
            processor.tokenizer.unk_token_id,
            processor.tokenizer.word_delimiter_token,
        )
        return processor

    # ------------------------------------------------------------------
    # Slow path (first run): build vocabulary + create processor
    # ------------------------------------------------------------------
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(
            f"Training transcript CSV not found at: {train_csv_path}\n"
            "Cannot build vocabulary without transcripts. "
            "Run preprocess.py first."
        )

    # 1. Build vocabulary from transcripts
    vocab = _build_vocabulary(train_csv_path)

    # 2. Write vocab.json (only if not already present)
    vocab_path = os.path.join(processor_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        with open(vocab_path, "w", encoding="utf-8") as fh:
            json.dump(vocab, fh, ensure_ascii=False, indent=2)
        logger.info("vocab.json written to: %s", vocab_path)
    else:
        logger.info("vocab.json already exists, skipping write: %s", vocab_path)

    # 3. Create Wav2Vec2CTCTokenizer from the local vocab file
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=vocab_path,
        unk_token=_UNK_TOKEN,
        pad_token=_PAD_TOKEN,
        word_delimiter_token=_WORD_DELIMITER_TOKEN,
    )

    # 4. Create Wav2Vec2FeatureExtractor
    #    Configured for mono, 16 kHz audio with zero-mean normalisation,
    #    zero padding value, and attention mask enabled — matching the
    #    existing preprocessor_config.json in the pretrained directory.
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    # 5. Combine into Wav2Vec2Processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # 6. Save processor so future runs use the fast path
    logger.info("Saving processor to: %s", processor_dir)
    processor.save_pretrained(processor_dir)
    logger.info(
        "Processor saved. Files written: %s",
        ", ".join(_TOKENIZER_SENTINEL_FILES),
    )

    return processor
