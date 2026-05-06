#!/usr/bin/env python3
"""Validate and inspect a RAGAS evaluation dataset.

Usage:
    python scripts/build_ragas_dataset.py --validate tests/fixtures/ragas_dataset.jsonl
    python scripts/build_ragas_dataset.py --validate tests/fixtures/ragas_dataset.jsonl --stats
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {"id", "question", "ground_truth"}
OPTIONAL_FIELDS = {"answer", "contexts", "metadata"}


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

            if not isinstance(row["question"], str) or not row["question"].strip():
                errors.append(f"  Line {line_no}: question is empty")
            if not isinstance(row["ground_truth"], str) or not row["ground_truth"].strip():
                errors.append(f"  Line {line_no}: ground_truth is empty")

            answer = row.get("answer")
            if answer is not None and not isinstance(answer, str):
                errors.append(f"  Line {line_no}: answer must be a string")

            contexts = row.get("contexts")
            if contexts is not None:
                if not isinstance(contexts, list):
                    errors.append(f"  Line {line_no}: contexts must be a list")
                else:
                    for idx, context in enumerate(contexts, start=1):
                        if not isinstance(context, dict):
                            errors.append(
                                f"  Line {line_no}: contexts[{idx}] must be an object"
                            )
                            continue
                        if not isinstance(context.get("text"), str) or not context["text"].strip():
                            errors.append(
                                f"  Line {line_no}: contexts[{idx}].text must be a non-empty string"
                            )
                        is_relevant = context.get("is_relevant")
                        if is_relevant is not None and not isinstance(is_relevant, bool):
                            errors.append(
                                f"  Line {line_no}: contexts[{idx}].is_relevant must be a boolean"
                            )

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
    has_contexts = 0
    relevant_context_total = 0

    for row in rows:
        meta = row.get("metadata", {})
        categories[meta.get("category", "uncategorised")] += 1
        difficulties[meta.get("difficulty", "unspecified")] += 1
        languages[meta.get("language", "unknown")] += 1
        contexts = row.get("contexts") or []
        if contexts:
            has_contexts += 1
            relevant_context_total += sum(
                1 for context in contexts if isinstance(context, dict) and context.get("is_relevant", True)
            )

    print(f"\n--- Dataset Statistics ({len(rows)} samples) ---")
    print(f"  Samples with contexts: {has_contexts}/{len(rows)}")
    print(f"  Relevant contexts total: {relevant_context_total}")

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
