"""Shared error-rate metric utilities for evaluation tooling."""

from __future__ import annotations

from statistics import fmean
from typing import Iterable, Mapping, Sequence

import evaluate

WER_METRIC = evaluate.load("wer")
CER_METRIC = evaluate.load("cer")


def compute_wer(references: Sequence[str], predictions: Sequence[str]) -> float:
    """Return the word error rate for the provided sequences."""
    if not references or not predictions:
        return 1.0
    return float(WER_METRIC.compute(references=references, predictions=predictions))


def compute_cer(references: Sequence[str], predictions: Sequence[str]) -> float:
    """Return the character error rate for the provided sequences."""
    if not references or not predictions:
        return 1.0
    return float(CER_METRIC.compute(references=references, predictions=predictions))


def summarize_error_rates(
    references: Sequence[str],
    predictions: Sequence[str],
    timings: Iterable[float] | None = None,
    *,
    failures: int = 0,
) -> Mapping[str, float | int]:
    """Compute aggregate WER/CER/timing statistics for a batch."""
    if not references or not predictions:
        return {
            "wer": 1.0,
            "cer": 1.0,
            "timing": 0.0,
            "n_samples": len(predictions),
            "failures": failures,
        }

    timings_list = list(timings) if timings is not None else []

    return {
        "wer": compute_wer(references, predictions),
        "cer": compute_cer(references, predictions),
        "timing": fmean(timings_list) if timings_list else 0.0,
        "n_samples": len(predictions),
        "failures": failures,
    }


__all__ = ["compute_wer", "compute_cer", "summarize_error_rates"]
