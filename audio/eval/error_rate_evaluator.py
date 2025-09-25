"""Utilities for running WER/CER evaluations across multiple ASR services."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
)

import evaluate

from .common_voice_dataset import CommonVoiceDataset
from .text_normalizer_utils import DEFAULT_NORMALIZER, Normalizer

try:  # Optional progress bar support
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

ServiceFuncMap = Mapping[str, Callable[[str, Optional[str]], str]]
NormalizerResolver = Callable[[str], Normalizer]

WER_METRIC = evaluate.load("wer")
CER_METRIC = evaluate.load("cer")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> MutableMapping[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _save_json(path: Path, payload: MutableMapping[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _record_transcript(path: Optional[Path], record: Dict[str, Any]) -> None:
    if not path:
        return
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _setup_logger(log_file: Path) -> logging.Logger:
    _ensure_parent(log_file)
    logger = logging.getLogger("error_rate_evaluator")
    logger.setLevel(logging.INFO)

    # Avoid attaching duplicate handlers when rerunning within notebooks.
    file_handler_exists = any(
        isinstance(handler, logging.FileHandler)
        and getattr(handler, "baseFilename", None) == str(log_file)
        for handler in logger.handlers
    )
    if not file_handler_exists:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    stream_handler_exists = any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    )
    if not stream_handler_exists:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(stream_handler)

    return logger


def _progress_iter(
    iterable,
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    enabled: bool = False,
):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def _compute_wer(references: List[str], predictions: List[str]) -> float:
    if not references or not predictions:
        return 1.0
    # The HF evaluate WER matches the OpenASR reference implementation
    return float(WER_METRIC.compute(references=references, predictions=predictions))


def _compute_cer(references: List[str], predictions: List[str]) -> float:
    if not references or not predictions:
        return 1.0
    return float(CER_METRIC.compute(references=references, predictions=predictions))


def _transcribe_with_retry(
    service_name: str,
    audio_path: str,
    language: Optional[str],
    transcribe_func,
    max_retries: int,
    logger: logging.Logger,
    lang_code: str,
    sample_idx: int,
) -> Dict[str, Any]:
    start_time = time.time()

    for attempt in range(max_retries):
        try:
            transcription = transcribe_func(audio_path, language)
            end_time = time.time()
            return {
                "service": service_name,
                "transcription": transcription,
                "timing": end_time - start_time,
                "success": True,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Attempt %s failed for %s on %s sample %s: %s",
                attempt + 1,
                service_name,
                lang_code,
                sample_idx,
                exc,
            )
            if attempt == max_retries - 1:
                logger.error(
                    "Failed to transcribe %s sample %s after %s attempts",
                    lang_code,
                    sample_idx,
                    max_retries,
                )
                end_time = time.time()
                return {
                    "service": service_name,
                    "transcription": "",
                    "timing": end_time - start_time,
                    "success": False,
                }

    # Should never reach here due to return statements above.
    return {
        "service": service_name,
        "transcription": "",
        "timing": 0.0,
        "success": False,
    }


def run_wer_evaluation(
    dataset_path: str,
    services: Iterable[str],
    service_funcs: ServiceFuncMap,
    languages: Iterable[str],
    results_file: str,
    checkpoint_file: str,
    log_file: str,
    *,
    max_retries: int = 3,
    n_samples: int = 100,
    split: Optional[str] = None,
    default_split: str = "test_100",
    normalizer_resolver: Optional[NormalizerResolver] = None,
    show_progress: bool = False,
    transcript_dump: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run a sequential WER/CER evaluation across services."""

    log_path = Path(log_file)
    results_path = Path(results_file)
    checkpoint_path = Path(checkpoint_file)

    logger = _setup_logger(log_path)

    dataset = CommonVoiceDataset(dataset_path, default_split=default_split)
    target_split = split or dataset.default_split

    resolved_normalizer = normalizer_resolver or (lambda lang: DEFAULT_NORMALIZER)

    checkpoint = _load_json(checkpoint_path)
    completed_services = set(checkpoint.get("completed", []))

    results: Dict[str, Dict[str, Any]] = {}

    dump_path = Path(transcript_dump) if transcript_dump else None

    languages_list = list(languages)
    for lang_code in _progress_iter(
        languages_list,
        total=len(languages_list),
        desc="Languages",
        enabled=show_progress,
    ):
        if lang_code not in dataset.languages:
            logger.warning("Language %s not found in dataset", lang_code)
            continue

        normalizer: Normalizer = resolved_normalizer(lang_code)
        logger.info("Processing language: %s", lang_code)

        samples_df = dataset.get_samples(
            lang_code, n_samples=n_samples, split=target_split
        )
        if len(samples_df) == 0:
            logger.warning("No samples found for %s", lang_code)
            continue

        lang_results: Dict[str, Any] = {}

        for service_name in services:
            if service_name not in service_funcs:
                raise KeyError(
                    f"Service '{service_name}' missing from service_funcs mapping"
                )

            checkpoint_key = f"{lang_code}_{service_name}"
            if checkpoint_key in completed_services:
                logger.info("Skipping completed: %s", checkpoint_key)
                continue

            transcribe_func = service_funcs[service_name]
            predictions: List[str] = []
            references: List[str] = []
            timings: List[float] = []

            rows = list(samples_df.iterrows())
            for idx, (_, row) in enumerate(
                _progress_iter(
                    rows,
                    total=len(rows),
                    desc=f"{lang_code}:{service_name}",
                    enabled=show_progress,
                )
            ):
                sample = dataset.get_sample_with_audio(lang_code, row)
                raw_text = sample["text"]
                normalized_reference = normalizer(raw_text)
                references.append(normalized_reference)

                result = _transcribe_with_retry(
                    service_name,
                    sample["audio_path"],
                    lang_code,
                    transcribe_func,
                    max_retries,
                    logger,
                    lang_code,
                    idx,
                )
                timings.append(result["timing"])
                transcription = result["transcription"]
                normalized_prediction = (
                    normalizer(transcription) if transcription else ""
                )
                predictions.append(normalized_prediction)

                _record_transcript(
                    dump_path,
                    {
                        "lang": lang_code,
                        "service": service_name,
                        "sample_index": idx,
                        "audio_path": sample["audio_path"],
                        "reference": raw_text,
                        "normalized_reference": normalized_reference,
                        "transcription": transcription,
                        "normalized_transcription": normalized_prediction,
                        "timing": result["timing"],
                        "success": result["success"],
                    },
                )

                if (idx + 1) % 10 == 0:
                    logger.info(
                        "Processed %s/%s samples for %s",
                        idx + 1,
                        len(samples_df),
                        service_name,
                    )

            if predictions and references:
                wer = _compute_wer(references, predictions)
                cer = _compute_cer(references, predictions)
                avg_time = sum(timings) / len(timings)
            else:
                wer = 1.0
                cer = 1.0
                avg_time = 0.0

            lang_results[service_name] = {
                "wer": wer,
                "cer": cer,
                "timing": avg_time,
                "n_samples": len(predictions),
            }
            completed_services.add(checkpoint_key)

            _save_json(checkpoint_path, {"completed": list(completed_services)})
            _save_json(results_path, {**results, lang_code: lang_results})

        if lang_results:
            results[lang_code] = lang_results

    logger.info("Evaluation completed")
    return results


def run_wer_evaluation_parallel(
    dataset_path: str,
    services: Iterable[str],
    service_funcs: ServiceFuncMap,
    languages: Iterable[str],
    results_file: str,
    checkpoint_file: str,
    log_file: str,
    *,
    max_retries: int = 3,
    n_samples: int = 100,
    max_workers: int = 8,
    split: Optional[str] = None,
    default_split: str = "test_100",
    normalizer_resolver: Optional[NormalizerResolver] = None,
    show_progress: bool = False,
    transcript_dump: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run a WER/CER evaluation using parallel service calls per sample."""

    log_path = Path(log_file)
    results_path = Path(results_file)
    checkpoint_path = Path(checkpoint_file)

    logger = _setup_logger(log_path)

    dataset = CommonVoiceDataset(dataset_path, default_split=default_split)
    target_split = split or dataset.default_split

    resolved_normalizer = normalizer_resolver or (lambda lang: DEFAULT_NORMALIZER)

    checkpoint = _load_json(checkpoint_path)
    completed_services = set(checkpoint.get("completed", []))

    existing_results = _load_json(results_path)
    results: Dict[str, Dict[str, Any]] = (
        {k: v for k, v in existing_results.items()} if existing_results else {}
    )

    dump_path = Path(transcript_dump) if transcript_dump else None

    languages_list = list(languages)
    for lang_code in _progress_iter(
        languages_list,
        total=len(languages_list),
        desc="Languages",
        enabled=show_progress,
    ):
        if lang_code not in dataset.languages:
            logger.warning("Language %s not found in dataset", lang_code)
            continue

        normalizer: Normalizer = resolved_normalizer(lang_code)
        logger.info("Processing language: %s", lang_code)

        samples_df = dataset.get_samples(
            lang_code, n_samples=n_samples, split=target_split
        )
        if len(samples_df) == 0:
            logger.warning("No samples found for %s", lang_code)
            continue

        services_to_process = [
            s for s in services if f"{lang_code}_{s}" not in completed_services
        ]
        lang_results: Dict[str, Any] = results.get(lang_code, {})

        if not services_to_process:
            logger.info("All services already completed for %s", lang_code)
            results[lang_code] = lang_results
            continue

        logger.info("Evaluating %s services on %s", len(services_to_process), lang_code)

        all_predictions = {service: [] for service in services_to_process}
        all_timings = {service: [] for service in services_to_process}
        all_references: List[str] = []

        rows = list(samples_df.iterrows())
        for idx, (_, row) in enumerate(
            _progress_iter(
                rows,
                total=len(rows),
                desc=f"{lang_code} samples",
                enabled=show_progress,
            )
        ):
            sample = dataset.get_sample_with_audio(lang_code, row)
            raw_text = sample["text"]
            normalized_reference = normalizer(raw_text)
            all_references.append(normalized_reference)
            audio_path = sample["audio_path"]

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = {
                    executor.submit(
                        _transcribe_with_retry,
                        service_name,
                        audio_path,
                        lang_code,
                        service_funcs[service_name],
                        max_retries,
                        logger,
                        lang_code,
                        idx,
                    ): service_name
                    for service_name in services_to_process
                }

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    service_name = result["service"]
                    transcription = result["transcription"]
                    normalized_transcription = (
                        normalizer(transcription) if transcription else ""
                    )
                    all_predictions[service_name].append(normalized_transcription)
                    all_timings[service_name].append(result["timing"])

                    _record_transcript(
                        dump_path,
                        {
                            "lang": lang_code,
                            "service": service_name,
                            "sample_index": idx,
                            "audio_path": audio_path,
                            "reference": raw_text,
                            "normalized_reference": normalized_reference,
                            "transcription": transcription,
                            "normalized_transcription": normalized_transcription,
                            "timing": result["timing"],
                            "success": result["success"],
                        },
                    )

            if (idx + 1) % 5 == 0:
                logger.info("Processed %s/%s samples", idx + 1, len(samples_df))

        for service_name in services_to_process:
            predictions = all_predictions[service_name]
            timings = all_timings[service_name]

            if predictions and all_references:
                wer = _compute_wer(all_references, predictions)
                cer = _compute_cer(all_references, predictions)
                avg_time = sum(timings) / len(timings)
            else:
                wer = 1.0
                cer = 1.0
                avg_time = 0.0

            lang_results[service_name] = {
                "wer": wer,
                "cer": cer,
                "timing": avg_time,
                "n_samples": len(predictions),
            }

            completed_services.add(f"{lang_code}_{service_name}")
            _save_json(checkpoint_path, {"completed": list(completed_services)})

        results[lang_code] = lang_results
        _save_json(results_path, results)

    logger.info("Evaluation completed")
    return results


__all__ = [
    "run_wer_evaluation",
    "run_wer_evaluation_parallel",
]
