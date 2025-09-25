#!/usr/bin/env python3
"""Report transcript success rates per language/service."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize transcript JSONL success counts"
    )
    parser.add_argument(
        "transcripts",
        type=Path,
        help="Path to transcript JSONL file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List individual failing rows",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {text}") from exc
    return records


def summarize(records: list[dict]) -> tuple[dict, list[tuple[dict, int]]]:
    summary = defaultdict(lambda: {"success": 0, "failed": 0})
    failures: list[tuple[dict, int]] = []
    for idx, record in enumerate(records):
        key = (record.get("lang", "unknown"), record.get("service", "unknown"))
        bucket = summary[key]
        if record.get("success", True):
            bucket["success"] += 1
        else:
            bucket["failed"] += 1
            failures.append((record, idx))
    return summary, failures


def format_summary(
    summary: dict, failures: list[tuple[dict, int]], verbose: bool
) -> str:
    lines = ["=== Transcript Status ==="]
    needs_attention = [k for k, v in summary.items() if v["failed"]]

    def sort_key(item: tuple[tuple[str, str], dict[str, int]]) -> tuple[int, str, str]:
        (lang, service), counts = item
        return (0 if counts["failed"] else 1, lang, service)

    for (lang, service), counts in sorted(summary.items(), key=sort_key):
        icon = "✅" if counts["failed"] == 0 else "⚠️"
        lines.append(
            f"{icon} {lang} | {service}: success={counts['success']} failed={counts['failed']}"
        )
    if needs_attention:
        lines.append("\n⚠️ Services requiring rerun:")
        for lang, service in sorted(needs_attention):
            lines.append(f"- {lang} | {service}")
    else:
        lines.append("\nAll services succeeded. ✅")

    if verbose and failures:
        lines.append("\n⚠️ Failed entries:")
        for record, idx in failures:
            lines.append(
                f"[{idx}] {record.get('lang')} | {record.get('service')} | sample={record.get('sample_index')}"
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not args.transcripts.exists():
        raise FileNotFoundError(f"Transcript file not found: {args.transcripts}")
    records = read_records(args.transcripts)
    summary, failures = summarize(records)
    report = format_summary(summary, failures, verbose=args.verbose)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
