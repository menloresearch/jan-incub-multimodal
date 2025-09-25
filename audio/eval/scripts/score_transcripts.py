#!/usr/bin/env python3
"""Compute WER/CER from cached transcript dumps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List, Tuple

import evaluate

WER_METRIC = evaluate.load("wer")
CER_METRIC = evaluate.load("cer")

TranscriptKey = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute error rates from cached transcript JSONL files."
    )
    parser.add_argument(
        "transcripts",
        type=Path,
        help="Path to the transcript JSONL file produced by the evaluator",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        help="Optional list of language codes to include (default: all)",
    )
    parser.add_argument(
        "--services",
        nargs="*",
        help="Optional list of service identifiers to include (default: all)",
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use raw reference/transcription text instead of normalized fields",
    )
    return parser.parse_args()


def read_transcripts(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON on line {line_number}: {text}") from exc
    return records


def filter_records(
    records: Iterable[dict],
    languages: Iterable[str] | None,
    services: Iterable[str] | None,
) -> List[dict]:
    languages_set = set(languages) if languages else None
    services_set = set(services) if services else None

    filtered: List[dict] = []
    for record in records:
        if languages_set and record.get("lang") not in languages_set:
            continue
        if services_set and record.get("service") not in services_set:
            continue
        filtered.append(record)
    return filtered


def compute_error_rates(
    records: Iterable[dict], *, use_raw: bool = False
) -> tuple[dict, dict | None]:
    grouped: DefaultDict[TranscriptKey, List[dict]] = defaultdict(list)
    for record in records:
        key = (record.get("lang", "unknown"), record.get("service", "unknown"))
        grouped[key].append(record)

    results = {}
    overall_refs: List[str] = []
    overall_preds: List[str] = []

    for key, group in grouped.items():
        references: List[str] = []
        predictions: List[str] = []
        failures = 0

        for item in group:
            ref_field = "reference" if use_raw else "normalized_reference"
            hyp_field = "transcription" if use_raw else "normalized_transcription"

            reference = item.get(ref_field) or ""
            prediction = item.get(hyp_field) or ""

            if not reference:
                continue

            references.append(reference)
            predictions.append(prediction)
            if not item.get("success", True):
                failures += 1

        if not references:
            continue

        wer = float(WER_METRIC.compute(references=references, predictions=predictions))
        cer = float(CER_METRIC.compute(references=references, predictions=predictions))

        results[key] = {
            "wer": wer,
            "cer": cer,
            "samples": len(references),
            "failures": failures,
        }

        overall_refs.extend(references)
        overall_preds.extend(predictions)

    overall = None
    if overall_refs:
        overall = {
            "wer": float(
                WER_METRIC.compute(references=overall_refs, predictions=overall_preds)
            ),
            "cer": float(
                CER_METRIC.compute(references=overall_refs, predictions=overall_preds)
            ),
            "samples": len(overall_refs),
        }

    return results, overall


def print_results(results: dict, overall: dict | None) -> None:
    if not results:
        print("No matching transcript entries found.")
        return

    print("=== Transcript Error Rates ===")
    for (language, service), metrics in sorted(results.items()):
        wer = metrics["wer"] * 100
        cer = metrics["cer"] * 100
        samples = metrics["samples"]
        failures = metrics["failures"]
        failure_note = f", failures={failures}" if failures else ""
        print(
            f"{language} | {service}: WER={wer:.2f}% CER={cer:.2f}% samples={samples}{failure_note}"
        )
    if overall:
        print("\n---")
        print(
            f"Overall: WER={overall['wer'] * 100:.2f}% CER={overall['cer'] * 100:.2f}% samples={overall['samples']}"
        )


def main() -> int:
    args = parse_args()
    records = read_transcripts(args.transcripts)
    filtered = filter_records(records, args.languages, args.services)
    results, overall = compute_error_rates(filtered, use_raw=args.use_raw)
    print_results(results, overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
