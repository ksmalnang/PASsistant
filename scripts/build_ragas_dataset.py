#!/usr/bin/env python3
"""Validate and inspect a RAGAS evaluation dataset.

Usage:
    python scripts/build_ragas_dataset.py --validate tests/fixtures/ragas_eval_dataset.jsonl
    python scripts/build_ragas_dataset.py --validate tests/fixtures/ragas_eval_dataset.jsonl --stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {"id", "user_input", "reference"}
OPTIONAL_FIELDS = {"reference_contexts", "metadata"}


def validate_dataset(path: Path) -> list[dict]:
    """Validate every line of a JSONL dataset and return parsed rows.

    Raises SystemExit on validation failure.
    """
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    errors: list[str] = []

    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"  Line {line_no}: invalid JSON — {exc}")
                continue

            missing = REQUIRED_FIELDS - set(row.keys())
            if missing:
                errors.append(
                    f"  Line {line_no} (id={row.get('id', '?')}): "
                    f"missing field(s): {', '.join(sorted(missing))}"
                )
                continue

            if not isinstance(row["user_input"], str) or not row["user_input"].strip():
                errors.append(f"  Line {line_no}: user_input is empty")
            if not isinstance(row["reference"], str) or not row["reference"].strip():
                errors.append(f"  Line {line_no}: reference is empty")

            ref_ctx = row.get("reference_contexts")
            if ref_ctx is not None and not isinstance(ref_ctx, list):
                errors.append(f"  Line {line_no}: reference_contexts must be a list")

            rows.append(row)

    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(err, file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("VALIDATION FAILED — dataset is empty", file=sys.stderr)
        sys.exit(1)

    print(f"OK: Dataset is valid ({len(rows)} samples)")
    return rows


def print_stats(rows: list[dict]) -> None:
    """Print summary statistics for the dataset."""
    categories: Counter[str] = Counter()
    difficulties: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    has_ref_ctx = 0

    for row in rows:
        meta = row.get("metadata", {})
        categories[meta.get("category", "uncategorised")] += 1
        difficulties[meta.get("difficulty", "unspecified")] += 1
        languages[meta.get("language", "unknown")] += 1
        if row.get("reference_contexts"):
            has_ref_ctx += 1

    print(f"\n--- Dataset Statistics ({len(rows)} samples) ---")
    print(f"  Samples with reference_contexts: {has_ref_ctx}/{len(rows)}")

    print("\n  Categories:")
    for cat, count in categories.most_common():
        print(f"    {cat:30s}  {count}")

    print("\n  Difficulty levels:")
    for diff, count in difficulties.most_common():
        print(f"    {diff:30s}  {count}")

    print("\n  Languages:")
    for lang, count in languages.most_common():
        print(f"    {lang:30s}  {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate / inspect RAGAS eval dataset")
    parser.add_argument(
        "--validate",
        required=True,
        help="Path to JSONL dataset to validate",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics after validation",
    )
    args = parser.parse_args()

    rows = validate_dataset(Path(args.validate))
    if args.stats:
        print_stats(rows)


if __name__ == "__main__":
    main()
