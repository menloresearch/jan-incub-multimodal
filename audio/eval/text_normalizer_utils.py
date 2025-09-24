"""Helpers for selecting language-specific text normalizers."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from .normalizer import BasicMultilingualTextNormalizer, EnglishTextNormalizer

Normalizer = Callable[[str], str]

english_normalizer = EnglishTextNormalizer()
multilingual_normalizer = BasicMultilingualTextNormalizer()
DEFAULT_NORMALIZER: Normalizer = english_normalizer


@lru_cache(maxsize=None)
def get_text_normalizer(lang_code: str) -> Normalizer:
    """Return a normalizer suited for the provided language code."""
    if lang_code is None:
        return DEFAULT_NORMALIZER

    normalized_code = lang_code.lower()
    if normalized_code in {"en", "eng", "english"}:
        return english_normalizer
    return multilingual_normalizer


__all__ = [
    "Normalizer",
    "DEFAULT_NORMALIZER",
    "english_normalizer",
    "multilingual_normalizer",
    "get_text_normalizer",
]
