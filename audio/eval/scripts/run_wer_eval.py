#!/usr/bin/env python3
"""CLI to run Common Voice WER/CER evaluations outside the notebook."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable
from typing import Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIO_EVAL_ROOT = REPO_ROOT / "audio" / "eval"
for candidate in (REPO_ROOT, AUDIO_EVAL_ROOT):
    if str(candidate) not in sys.path:
        sys.path.append(str(candidate))

NOTEBOOKS_DIR = AUDIO_EVAL_ROOT / "notebooks"

from audio.eval import (  # noqa: E402
    DEFAULT_SERVICE_MODELS,
    DEFAULT_SERVICES,
    CommonVoiceDataset,
    build_service_function_map,
    get_text_normalizer,
    run_wer_evaluation,
    run_wer_evaluation_parallel,
)

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "results"

PRESET_LANGUAGE_SETS = {
    "smoke": ["en"],
    "major": ["en", "es", "fr", "de", "it", "ja", "pt", "ru"],
    "all": None,  # resolved dynamically to the full dataset
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WER/CER evaluation across ASR services"
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to the Common Voice dataset root (fallback to $CV22_PATH)",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Language codes to evaluate (default: all languages present)",
    )
    parser.add_argument(
        "--languages-file",
        type=Path,
        help="Optional file containing one language code per line",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_LANGUAGE_SETS.keys()),
        help="Named language set to evaluate (overridden by --languages or --languages-file)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to read (default: dataset default, typically 'test_100')",
    )
    parser.add_argument(
        "--default-split",
        default="test_100",
        help="Default split used when none is provided and for dataset sampling",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per language",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per service call",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Thread pool size for parallel evaluation",
    )
    parser.add_argument(
        "--services",
        nargs="*",
        default=None,
        help=f"Service identifiers to run (default: all, choices include: {', '.join(DEFAULT_SERVICES)})",
    )
    parser.add_argument(
        "--service-model",
        action="append",
        metavar="SERVICE:MODEL",
        help="Override or add a service/model pair (e.g. vllm:large-v3)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to write the aggregated results JSON (default: results/wer_results_<ts>.json)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to write checkpoint progress (default: results/wer_checkpoint_<ts>.json)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Path to write the evaluation log (default: logs/wer_eval_<ts>.log)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run services sequentially instead of in parallel",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        action="append",
        help="Additional .env file(s) to load for API keys",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable progress bars",
    )
    parser.set_defaults(show_progress=True)
    parser.add_argument(
        "--dump-transcripts",
        type=Path,
        help="JSONL file to store raw and normalized transcripts (default: logs/transcripts_<timestamp>.jsonl)",
    )
    return parser.parse_args()


def load_environment(extra_env_files: Optional[Iterable[Path]]) -> None:
    load_dotenv()
    notebooks_env = NOTEBOOKS_DIR / ".env"
    if notebooks_env.exists():
        load_dotenv(notebooks_env)
    if extra_env_files:
        for env_path in extra_env_files:
            load_dotenv(env_path, override=True)


def read_languages(dataset: CommonVoiceDataset, args: argparse.Namespace) -> list[str]:
    if args.languages_file:
        contents = args.languages_file.read_text(encoding="utf-8").splitlines()
        languages = [
            line.strip()
            for line in contents
            if line.strip() and not line.strip().startswith("#")
        ]
        if languages:
            return languages
    if args.languages:
        return list(args.languages)
    if args.preset:
        preset = PRESET_LANGUAGE_SETS[args.preset]
        if preset is None:
            return dataset.languages
        return preset
    return dataset.languages


def parse_service_overrides(
    overrides: Optional[Iterable[str]],
) -> list[tuple[str, str]]:
    if not overrides:
        return []
    pairs: list[tuple[str, str]] = []
    for item in overrides:
        if ":" not in item:
            raise ValueError(
                f"Invalid service override '{item}' (expected SERVICE:MODEL)"
            )
        service, model = item.split(":", 1)
        if not service or not model:
            raise ValueError(f"Invalid service override '{item}'")
        pairs.append((service.strip(), model.strip()))
    return pairs


def resolve_services(args: argparse.Namespace) -> dict[str, callable]:
    service_models = list(DEFAULT_SERVICE_MODELS)
    service_models.extend(parse_service_overrides(args.service_model))
    service_funcs = build_service_function_map(service_models)

    if args.services:
        missing = sorted(set(args.services) - set(service_funcs))
        if missing:
            raise ValueError(f"Unknown services requested: {missing}")
        service_funcs = {key: service_funcs[key] for key in args.services}

    return service_funcs


def main() -> int:
    args = parse_args()
    load_environment(args.env_file)

    dataset_root = args.dataset_path or os.getenv("CV22_PATH")
    if not dataset_root:
        print(
            "Error: dataset path not provided and $CV22_PATH not set", file=sys.stderr
        )
        return 1

    dataset_root_path = Path(dataset_root).expanduser().resolve()
    if not dataset_root_path.exists():
        print(f"Error: dataset path not found: {dataset_root_path}", file=sys.stderr)
        return 1

    service_funcs = resolve_services(args)
    services = list(service_funcs.keys())

    dataset = CommonVoiceDataset(
        str(dataset_root_path), default_split=args.default_split
    )
    languages = read_languages(dataset, args)
    target_split = args.split or dataset.default_split

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = args.results or (
        DEFAULT_RESULTS_DIR / f"wer_results_{timestamp}.json"
    )
    checkpoint_path = args.checkpoint or (
        DEFAULT_CHECKPOINT_DIR / f"wer_checkpoint_{timestamp}.json"
    )
    log_path = args.log or (DEFAULT_LOGS_DIR / f"wer_eval_{timestamp}.log")
    transcript_path = args.dump_transcripts or (
        DEFAULT_LOGS_DIR / f"transcripts_{timestamp}.jsonl"
    )

    for path in (results_path, checkpoint_path, log_path, transcript_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    eval_kwargs = dict(
        dataset_path=str(dataset_root_path),
        services=services,
        service_funcs=service_funcs,
        languages=languages,
        results_file=str(results_path),
        checkpoint_file=str(checkpoint_path),
        log_file=str(log_path),
        max_retries=args.max_retries,
        n_samples=args.n_samples,
        split=target_split,
        default_split=args.default_split,
        normalizer_resolver=get_text_normalizer,
        transcript_dump=str(transcript_path),
        show_progress=args.show_progress,
    )

    print("=== Evaluation Configuration ===")
    print(f"Dataset root: {dataset_root_path}")
    print(f"Languages ({len(languages)}): {languages}")
    print(f"Services ({len(services)}): {services}")
    print(f"Split: {target_split} (default: {args.default_split})")
    print(f"Samples per language: {args.n_samples}")
    print(f"Retries per service call: {args.max_retries}")
    print(f"Results file: {results_path}")
    print(f"Checkpoint file: {checkpoint_path}")
    print(f"Log file: {log_path}")
    print(f"Transcript dump: {transcript_path}")
    print(f"Progress bar: {'on' if args.show_progress else 'off'}")
    print(f"Mode: {'sequential' if args.sequential else 'parallel'}")

    if args.sequential:
        results = run_wer_evaluation(**eval_kwargs)
    else:
        eval_kwargs["max_workers"] = args.max_workers
        results = run_wer_evaluation_parallel(**eval_kwargs)

    print("\n=== Evaluation Summary ===")
    for lang, lang_results in results.items():
        print(f"{lang}:")
        for service, metrics in lang_results.items():
            wer = metrics.get("wer", 1.0)
            cer = metrics.get("cer")
            timing = metrics.get("timing", 0.0)
            count = metrics.get("n_samples", 0)
            failures = int(metrics.get("failures", 0))
            cer_summary = f", CER={cer:.4f}" if cer is not None else ""
            failure_note = f", Failures={failures}" if failures else ""
            status_icon = "⚠️" if failures else "✅"
            print(
                f"  {status_icon} {service}: WER={wer:.4f}{cer_summary}, Avg Time={timing:.2f}s, Samples={count}{failure_note}"
            )
            if failures:
                print(
                    f"    ⚠️  {failures} failed sample(s) detected for {service}; rerun or inspect transcripts."
                )

    print(f"\nDone. Detailed results written to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
