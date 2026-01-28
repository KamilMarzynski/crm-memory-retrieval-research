"""
Export code review data from JSON files to CSV format.

This script reads all JSON files from data/review_data/ and exports them to a CSV
file with columns: context, file, severity, code, comment, user_note.

Each row represents one code review comment from the source PRs.

Usage:
    uv run python scripts/export_review_data_to_csv.py
    uv run python scripts/export_review_data_to_csv.py --output custom_output.csv
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from common import load_json

# Default paths
DEFAULT_INPUT_DIR = "data/review_data"
DEFAULT_OUTPUT_FILE = "data/review_data_export.csv"

# CSV column headers
CSV_HEADERS = ["context", "file", "severity", "code", "comment", "user_note"]


def extract_comments_from_pr(pr_file_path: str) -> List[Dict[str, str]]:
    """
    Extract code review comments from a PR JSON file.

    Args:
        pr_file_path: Path to raw PR JSON file.

    Returns:
        List of dictionaries, each representing a code review comment.
    """
    try:
        pr_data = load_json(pr_file_path)
    except Exception as e:
        print(f"Warning: Failed to load {pr_file_path}: {e}")
        return []

    pr_context = pr_data.get("context", "")

    comments = []
    for comment_data in pr_data.get("code_review_comments", []):
        comment = {
            "context": pr_context,
            "file": comment_data.get("file", ""),
            "severity": comment_data.get("severity", "info"),
            "code": comment_data.get("code_snippet", ""),
            "comment": comment_data.get("message", ""),
            "user_note": comment_data.get("user_note", ""),
        }
        comments.append(comment)

    return comments


def export_to_csv(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_file: str = DEFAULT_OUTPUT_FILE,
) -> None:
    """
    Export all code review comments from JSON files to CSV.

    Args:
        input_dir: Directory containing raw PR JSON files.
        output_file: Path where CSV file should be written.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)

    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Processing {input_dir}...")

    all_comments = []
    for json_file in json_files:
        comments = extract_comments_from_pr(str(json_file))
        all_comments.extend(comments)
        print(f"  {json_file.name}: {len(comments)} comments")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(all_comments)

    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Processed: {len(json_files)} JSON files")
    print(f"Exported: {len(all_comments)} code review comments")
    print(f"Output: {output_path}")
    print("=" * 60)
    print()
    print("CSV format:")
    print(f"  Columns: {', '.join(CSV_HEADERS)}")
    print("  Encoding: UTF-8")
    print()
    print("Ready to import into your tool!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("Export Code Review Data to CSV")
        print()
        print("Usage:")
        print("  uv run python scripts/export_review_data_to_csv.py")
        print("  uv run python scripts/export_review_data_to_csv.py --output custom.csv")
        print()
        print("Description:")
        print("  Reads all JSON files from data/review_data/ and exports code review")
        print("  comments to a CSV file with columns:")
        print()
        for header in CSV_HEADERS:
            print(f"    - {header}")
        print()
        print(f"Default output: {DEFAULT_OUTPUT_FILE}")
        sys.exit(0)

    if len(sys.argv) > 2 and sys.argv[1] == "--output":
        output_file = sys.argv[2]
        export_to_csv(output_file=output_file)
    else:
        export_to_csv()
