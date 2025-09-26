#!/usr/bin/env python3
"""Compute WER/CER from cached transcript dumps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIO_EVAL_ROOT = REPO_ROOT / "audio" / "eval"
for candidate in (REPO_ROOT, AUDIO_EVAL_ROOT):
    if str(candidate) not in sys.path:
        sys.path.append(str(candidate))

from audio.eval import get_text_normalizer  # noqa: E402
from audio.eval.metrics import compute_cer, compute_wer  # noqa: E402

TranscriptKey = tuple[str, str]


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
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the summary output (default: alongside input)",
    )
    return parser.parse_args()


def read_transcripts(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {path}")

    records: list[dict] = []
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
) -> list[dict]:
    languages_set = set(languages) if languages else None
    services_set = set(services) if services else None

    filtered: list[dict] = []
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
    grouped: defaultdict[TranscriptKey, list[dict]] = defaultdict(list)
    for record in records:
        key = (record.get("lang", "unknown"), record.get("service", "unknown"))
        grouped[key].append(record)

    results = {}
    overall_refs: list[str] = []
    overall_preds: list[str] = []

    normalizer_cache: dict[str, callable] = {}

    def normalize(lang: str, text: str) -> str:
        if not text:
            return ""
        normalizer = normalizer_cache.get(lang)
        if normalizer is None:
            normalizer = get_text_normalizer(lang)
            normalizer_cache[lang] = normalizer
        return normalizer(text)

    for key, group in grouped.items():
        references: list[str] = []
        predictions: list[str] = []
        failures = 0

        for item in group:
            reference_raw = item.get("reference") or ""
            prediction_raw = item.get("transcription") or ""

            if not reference_raw:
                continue

            if use_raw:
                references.append(reference_raw)
                predictions.append(prediction_raw)
            else:
                lang = item.get("lang", "unknown")
                references.append(normalize(lang, reference_raw))
                predictions.append(normalize(lang, prediction_raw))
            if not item.get("success", True):
                failures += 1

        if not references:
            continue

        wer = compute_wer(references, predictions)
        cer = compute_cer(references, predictions)

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
            "wer": compute_wer(overall_refs, overall_preds),
            "cer": compute_cer(overall_refs, overall_preds),
            "samples": len(overall_refs),
        }

    return results, overall


def format_results(results: dict, overall: dict | None) -> str:
    if not results:
        return "No matching transcript entries found."

    lines = ["=== Transcript Error Rates ==="]
    for (language, service), metrics in sorted(results.items()):
        wer = metrics["wer"] * 100
        cer = metrics["cer"] * 100
        samples = metrics["samples"]
        failures = metrics["failures"]
        failure_note = f", failures={failures}" if failures else ""
        lines.append(
            f"{language} | {service}: WER={wer:.2f}% CER={cer:.2f}% samples={samples}{failure_note}"
        )
    if overall:
        lines.extend(
            [
                "",
                "---",
                f"Overall: WER={overall['wer'] * 100:.2f}% CER={overall['cer'] * 100:.2f}% samples={overall['samples']}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    records = read_transcripts(args.transcripts)
    filtered = filter_records(records, args.languages, args.services)
    results, overall = compute_error_rates(filtered, use_raw=args.use_raw)
    summary = format_results(results, overall)
    print(summary)

    output_path = args.output
    if not output_path:
        output_name = f"{args.transcripts.stem}_summary.txt"
        output_path = args.transcripts.with_name(output_name)

    output_path.write_text(summary + "\n", encoding="utf-8")
    print(f"\nSaved summary to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
