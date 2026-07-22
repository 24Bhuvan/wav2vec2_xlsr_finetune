import csv
import json
from pathlib import Path
from typing import Any, Dict, List

# --- Configuration Constants & Absolute Path Construction ---
DEFAULT_THRESHOLD_MS: float = 70.0

BASE_DIR: Path = Path(__file__).resolve().parent
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

COMPARISON_REPORT_PATH: Path = OUTPUTS_DIR / "comparison_report.csv"
RED_FLAG_REPORT_PATH: Path = OUTPUTS_DIR / "red_flag_files.csv"
SUMMARY_JSON_PATH: Path = OUTPUTS_DIR / "summary.json"


def calculate_median(values: List[float]) -> float:
    """Computes the median value of a numerical sequence safely."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


def calculate_mean(values: List[float]) -> float:
    """Computes the arithmetic mean safely."""
    return sum(values) / len(values) if values else 0.0


def load_comparison_data(csv_path: Path) -> List[Dict[str, Any]]:
    """Parses raw text records from the comparison report into structured data."""
    records = []
    if not csv_path.is_file():
        return records

    try:
        with csv_path.open(mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if not row.get("file_id") or not row.get("status"):
                        continue
                    records.append(
                        {
                            "file_id": row["file_id"],
                            "status": row["status"],
                            "start_diff": float(row["start_diff"]),
                            "end_diff": float(row["end_diff"]),
                            "combined_diff": float(row["combined_diff"]),
                        }
                    )
                except (ValueError, KeyError):
                    continue  # Silently skip individual malformed records
    except Exception as exc:
        print(f"Error loading comparison data text file: {exc}")

    return records


def run_evaluation(threshold_ms: float = DEFAULT_THRESHOLD_MS) -> None:
    """Aggregates all word alignment drift records and outputs dataset-level statistics."""
    try:
        records = load_comparison_data(COMPARISON_REPORT_PATH)

        # Track discrete files via high-priority drift logs
        red_flag_files = set()
        all_files = set()

        if RED_FLAG_REPORT_PATH.is_file():
            with RED_FLAG_REPORT_PATH.open(mode="r", encoding="utf-8") as rf:
                r_reader = csv.DictReader(rf)
                for row in r_reader:
                    if row.get("file_id"):
                        red_flag_files.add(row["file_id"])

        # Primary telemetry tracking pools
        start_errors_ms: List[float] = []
        end_errors_ms: List[float] = []
        combined_errors_ms: List[float] = []

        correct_words = 0
        flagged_words = 0

        for r in records:
            all_files.add(r["file_id"])

            # Capture raw drift weights scaled to milliseconds
            s_err = r["start_diff"] * 1000.0
            e_err = r["end_diff"] * 1000.0
            c_err = r["combined_diff"] * 1000.0

            start_errors_ms.append(s_err)
            end_errors_ms.append(e_err)
            combined_errors_ms.append(c_err)

            if r["status"] == "Flagged":
                flagged_words += 1
            else:
                correct_words += 1

        total_words = correct_words + flagged_words
        files_compared = len(all_files)

        # Calculate percentages cleanly
        pct_correct = (correct_words / total_words * 100.0) if total_words > 0 else 0.0
        pct_flagged = (flagged_words / total_words * 100.0) if total_words > 0 else 0.0

        # 1. Compile metric definitions payload
        metrics = {
            "files_compared": files_compared,
            "missing_files": 0,  # Handled as an anomaly buffer or calculated externally
            "red_flag_files": len(red_flag_files),
            "total_words": total_words,
            "correct_words": correct_words,
            "flagged_words": flagged_words,
            "percentage_flagged_words": round(pct_flagged, 2),
            "percentage_correct_words": round(pct_correct, 2),
            "mean_start_error_ms": round(calculate_mean(start_errors_ms), 2),
            "mean_end_error_ms": round(calculate_mean(end_errors_ms), 2),
            "mean_combined_error_ms": round(calculate_mean(combined_errors_ms), 2),
            "median_combined_error_ms": round(calculate_median(combined_errors_ms), 2),
            "maximum_combined_error_ms": round(
                max(combined_errors_ms) if combined_errors_ms else 0.0, 2
            ),
            "minimum_combined_error_ms": round(
                min(combined_errors_ms) if combined_errors_ms else 0.0, 2
            ),
            "threshold_ms": threshold_ms,
        }

        # 2. Commit metadata block to persistent disk
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as j_out:
            json.dump(metrics, j_out, indent=4)

        # 3. Print the diagnostic dashboard to stdout
        print("=" * 50)
        print("Forced Alignment Evaluation Summary")
        print("=" * 50)
        print(f"Files Compared          : {metrics['files_compared']}")
        print(f"Missing Files           : {metrics['missing_files']}")
        print(f"Red Flag Files          : {metrics['red_flag_files']}")
        print("")
        print(f"Total Words             : {metrics['total_words']}")
        print(f"Correct Words           : {metrics['correct_words']}")
        print(f"Flagged Words           : {metrics['flagged_words']}")
        print("")
        print(f"Percentage Correct      : {metrics['percentage_correct_words']}%")
        print(f"Percentage Flagged      : {metrics['percentage_flagged_words']}%")
        print("")
        print(f"Mean Start Error        : {metrics['mean_start_error_ms']} ms")
        print(f"Mean End Error          : {metrics['mean_end_error_ms']} ms")
        print(f"Mean Combined Error     : {metrics['mean_combined_error_ms']} ms")
        print("")
        print(f"Median Error            : {metrics['median_combined_error_ms']} ms")
        print(f"Maximum Error           : {metrics['maximum_combined_error_ms']} ms")
        print("")
        print(f"Threshold               : {int(metrics['threshold_ms'])} ms")
        print("=" * 50)
        print("Evaluation Completed Successfully")
        print("=" * 50)

    except Exception as exc:
        print(f"Error executing forced alignment metrics evaluation: {exc}")


if __name__ == "__main__":
    run_evaluation()
