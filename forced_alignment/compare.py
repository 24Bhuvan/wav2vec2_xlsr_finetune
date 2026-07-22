import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

# Instantiate logger to stay consistent with the global infrastructure log files
logger = logging.getLogger("inference")


def parse_human_text_annotation(file_path: Path) -> List[Dict[str, Any]]:
    """Parses a plain text human annotation file.

    Expected format per line: <start_time> <end_time> <word>
    """
    parsed_words = []
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=2)
                if len(parts) < 3:
                    logger.warning(
                        "Malformed line %d in %s: Expected 3 elements, got %d. Skipping.",
                        line_idx,
                        file_path.name,
                        len(parts),
                    )
                    continue

                try:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    word = parts[2]
                    parsed_words.append({"start": start_time, "end": end_time, "word": word})
                except ValueError:
                    logger.warning(
                        "Value conversion failure at line %d in %s. Skipping line.",
                        line_idx,
                        file_path.name,
                    )
                    continue
    except Exception as exc:
        logger.error("Failed to parse human annotation text file %s: %s", file_path, exc)

    return parsed_words


def compare_annotations(
    predicted_dir: Union[str, Path] = "./outputs/predicted_annotations",
    human_dir: Union[str, Path] = "../data/raw/annotations",
    output_dir: Union[str, Path] = "./outputs",
    threshold_ms: float = 70.0,
) -> None:
    """Compares forced alignment word predictions against ground-truth human annotations.

    Parses space-separated text files for human annotations and processes predicted JSON blocks.
    """
    predicted_dir = Path(predicted_dir)
    human_dir = Path(human_dir)
    output_dir = Path(output_dir)

    threshold_sec = threshold_ms / 1000.0

    report_csv_path = output_dir / "comparison_report.csv"
    red_flag_csv_path = output_dir / "red_flag_files.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define fields for global tracking report
    report_headers = [
        "file_id",
        "word_index",
        "word",
        "human_start",
        "pred_start",
        "start_diff",
        "human_end",
        "pred_end",
        "end_diff",
        "combined_diff",
        "status",
    ]

    # Track files requiring immediate manual correction
    flagged_files_headers = ["file_id", "total_words", "flagged_words", "flagged_percentage"]

    flagged_files_summary: Dict[str, Dict[str, int]] = {}

    try:
        with report_csv_path.open(mode="w", newline="", encoding="utf-8") as rep_f:
            writer = csv.writer(rep_f)
            writer.writerow(report_headers)

            # Iterate over predicted JSON outputs
            if not predicted_dir.is_dir():
                logger.warning("Predictions directory not found: %s", predicted_dir)
                return

            for pred_file in predicted_dir.iterdir():
                if pred_file.suffix != ".json":
                    continue

                file_id = pred_file.stem

                # Resolve text-based human annotation following the dataset naming priority pattern
                human_path = human_dir / f"{file_id}_corrected.txt"
                if not human_path.is_file():
                    human_path = human_dir / f"{file_id}.txt"

                if not human_path.is_file():
                    msg = f"Human annotation missing for sample key: {file_id}"
                    print(msg)
                    logger.warning("%s (Checked *_corrected.txt and *.txt). Skipping.", msg)
                    continue

                try:
                    # Load predictions via JSON
                    with pred_file.open("r", encoding="utf-8") as pf:
                        pred_data = json.load(pf)

                    # Parse text human records manually line by line
                    human_data = parse_human_text_annotation(human_path)

                    flagged_words_count = 0
                    total_words = min(len(pred_data), len(human_data))

                    if total_words == 0:
                        logger.warning(
                            "Zero matching words alignment overlap for file token: %s", file_id
                        )
                        continue

                    flagged_files_summary[file_id] = {"total": total_words, "flagged": 0}

                    # Step through text structures word by word
                    for i in range(total_words):
                        p_word = pred_data[i]
                        h_word = human_data[i]

                        # Direct alignment metric calculation
                        start_diff = abs(p_word["start"] - h_word["start"])
                        end_diff = abs(p_word["end"] - h_word["end"])
                        combined_diff = start_diff + end_diff

                        # Check status against timing tolerances
                        if start_diff > threshold_sec or end_diff > threshold_sec:
                            status = "Flagged"
                            flagged_words_count += 1
                        else:
                            status = "Correct"

                        writer.writerow(
                            [
                                file_id,
                                i,
                                h_word["word"],
                                round(h_word["start"], 3),
                                round(p_word["start"], 3),
                                round(start_diff, 3),
                                round(h_word["end"], 3),
                                round(p_word["end"], 3),
                                round(end_diff, 3),
                                round(combined_diff, 3),
                                status,
                            ]
                        )

                    flagged_files_summary[file_id]["flagged"] = flagged_words_count

                except Exception as file_exc:
                    logger.error(
                        "Failed processing comparison execution for %s: %s",
                        file_id,
                        file_exc,
                        exc_info=True,
                    )
                    continue

        # Write out isolated action items for high error profiles
        with red_flag_csv_path.open(mode="w", newline="", encoding="utf-8") as red_f:
            red_writer = csv.writer(red_f)
            red_writer.writerow(flagged_files_headers)

            for fid, counts in flagged_files_summary.items():
                if counts["flagged"] > 0:
                    pct = round((counts["flagged"] / counts["total"]) * 100, 2)
                    red_writer.writerow([fid, counts["total"], counts["flagged"], pct])

        print("Comparison completed.")
        print(f" -> Comprehensive breakdown: {report_csv_path}")
        print(f" -> High-priority drift files: {red_flag_csv_path}")
        logger.info(
            "Comparison matrices generated successfully from text data. Reports saved to %s",
            output_dir,
        )

    except Exception as exc:
        logger.error("Failed to complete metric evaluation process: %s", exc, exc_info=True)
        print(f"Error executing annotation comparison: {exc}")


if __name__ == "__main__":
    compare_annotations()
