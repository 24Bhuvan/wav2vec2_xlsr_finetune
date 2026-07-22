import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger("inference")


def merge_tokens_to_words(
    alignment_path: List[Dict[str, Any]],
    token_ids: Union[List[int], Any],
    tokenizer: Any,
    frame_duration: float,
) -> List[Dict[str, Any]]:
    """Transforms a frame-level token alignment path into structured word-level
    timestamps by merging repeated characters and splitting on word boundaries.

    Args:
        alignment_path (list of dict): Viterbi path points from decoder.py
                                       [{"token_index": int, "frame_index": int, "score": float}, ...]
        token_ids (list): Sequence of target token IDs matching the transcript text.
        tokenizer: Initialized Hugging Face tokenizer mapping IDs back to strings.
        frame_duration (float): Exact duration of a single acoustic frame in seconds.

    Returns:
        list of dict: Clean word-level segments:
                      [{"word": str, "start": float, "end": float}, ...]
    """
    if not alignment_path:
        return []

    try:
        # Resolve vocabulary settings safely
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        word_delimiter = "|" if "|" in vocab else " "

        words = []
        current_word_chars = []

        # 1. Group frame allocations back to distinct character tokens
        char_segments = []
        last_transcript_idx = None

        for i, point in enumerate(alignment_path):
            token_idx = point.get("token_index", -1)
            frame_idx = point.get("frame_index", -1)

            # Ignore structural CTC blanks (standardly tagged with token_index = -1) or invalid indexes
            if token_idx < 0 or token_idx >= len(token_ids) or frame_idx < 0:
                continue

            token_id = token_ids[token_idx]
            
            # 1. Preserve raw tokenizer tokens and normalize immediately
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(token_str, list):
                token_str = token_str[0]
            token_str = token_str.strip()

            # 2. Automatically filter all tokenizer special tokens
            if token_str.startswith("<") and token_str.endswith(">"):
                continue

            # Explicitly ignore other standard blank/padding artifacts
            if token_str in ("[PAD]", ""):
                continue

            # Calculate time coordinates
            frame_start = frame_idx * frame_duration
            frame_end = frame_start + frame_duration

            # 1. Merge repeated frame assignments using an explicit tracking variable
            if last_transcript_idx != token_idx:
                char_segments.append({
                    "token_idx": token_idx,
                    "token": token_str,
                    "start": frame_start,
                    "end": frame_end,
                })
                last_transcript_idx = token_idx
            else:
                char_segments[-1]["end"] = frame_end

        # 2. Assemble localized characters into full text words splitting on delimiters
        for segment in char_segments:
            token_str = segment["token"]

            # Filter out padding strings or special zero-width tokens
            if not token_str.strip("\u200b"):
                continue

            # 4. Preserve word delimiters exactly as stored in the tokenizer vocabulary
            if token_str == word_delimiter:
                if current_word_chars:
                    words.append(current_word_chars)
                    current_word_chars = []
            else:
                current_word_chars.append(segment)

        if current_word_chars:
            words.append(current_word_chars)

        # 3. Format the groups into clean word boundaries rounded to milliseconds
        word_timestamps = []
        for word_group in words:
            full_word = "".join([c["token"] for c in word_group]).strip()

            if not full_word:
                continue

            start_time = word_group[0]["start"]
            end_time = word_group[-1]["end"]

            word_timestamps.append(
                {
                    "word": full_word,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                }
            )

        return word_timestamps

    except Exception as exc:
        logger.error(
            "Failed during token integration process to words: %s",
            exc,
            exc_info=True,
        )
        return []
