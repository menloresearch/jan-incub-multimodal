"""Utility class for managing Common Voice dataset splits and samples."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, Generator, Iterable, List, Optional

import numpy as np
import pandas as pd


class CommonVoiceDataset:
    """Helper for loading, filtering, and iterating through Common Voice samples."""

    _BASE_VALID_SPLITS: List[str] = [
        "train",
        "dev",
        "test",
        "validated",
        "invalidated",
        "other",
        "reported",
    ]

    def __init__(
        self,
        root_path: str,
        random_seed: int = 42,
        default_split: str = "test_100",
    ) -> None:
        self.root_path = os.path.abspath(os.path.realpath(os.path.expanduser(root_path)))
        self.languages = self._list_languages()
        self.random_seed = random_seed
        self.default_split = default_split
        self._valid_splits = list(self._BASE_VALID_SPLITS)
        if default_split not in self._valid_splits:
            self._valid_splits.append(default_split)

        print(f"Found {len(self.languages)} languages: {self.languages[:5]}...")

    # Private methods -----------------------------------------------------
    def __validate_language(self, lang_code: str) -> None:
        if lang_code not in self.languages:
            raise ValueError(f"Language '{lang_code}' not found")

    def __validate_split(self, split: str) -> None:
        if split not in self._valid_splits:
            raise ValueError(f"Invalid split '{split}'")

    # Protected helpers ---------------------------------------------------
    def _list_languages(self) -> List[str]:
        """List all language directories under the dataset root."""
        return [d for d in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, d))]

    def _get_language_path(self, lang_code: str) -> str:
        self.__validate_language(lang_code)
        return os.path.join(self.root_path, lang_code)

    def _get_split_tsv_path(self, lang_code: str, split: str) -> str:
        self.__validate_split(split)
        lang_path = self._get_language_path(lang_code)
        tsv_file = os.path.join(lang_path, f"{split}.tsv")

        if os.path.exists(tsv_file):
            return tsv_file

        if split == "test_100":
            logging.error(
                "Split %s not found for %s; ensure the subset exists or specify a different split",
                split,
                lang_code,
            )

        available = self._get_available_splits(lang_code)
        raise ValueError(f"Split '{split}' not found for {lang_code}. Available: {available}")

    def _get_available_splits(self, lang_code: str) -> List[str]:
        lang_path = self._get_language_path(lang_code)
        return [s for s in self._valid_splits if os.path.exists(os.path.join(lang_path, f"{s}.tsv"))]

    def _load_dataframe(self, tsv_path: str) -> pd.DataFrame:
        df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

        # Clean missing data columns if present.
        if "path" in df.columns and "sentence" in df.columns:
            df = df.dropna(subset=["path", "sentence"])
            df = df[df["sentence"].str.strip() != ""]

        return df

    def _apply_quality_filters(
        self,
        df: pd.DataFrame,
        min_up_votes: int = 0,
        require_gender: bool = False,
        require_age: bool = False,
    ) -> pd.DataFrame:
        if "up_votes" in df.columns and min_up_votes > 0:
            df = df[df["up_votes"] >= min_up_votes]

        if require_gender and "gender" in df.columns:
            df = df[df["gender"].notna()]

        if require_age and "age" in df.columns:
            df = df[df["age"].notna()]

        return df

    def _sample_deterministically(self, df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        np.random.seed(self.random_seed)

        if len(df) > n_samples:
            return df.sample(n=n_samples, random_state=self.random_seed)
        return df

    def _build_audio_path(self, lang_code: str, filename: str) -> str:
        lang_path = self._get_language_path(lang_code)

        # Try clips subdirectory first then fallback to lang root.
        audio_path = os.path.join(lang_path, "clips", filename)
        if os.path.exists(audio_path):
            return audio_path

        audio_path = os.path.join(lang_path, filename)
        if os.path.exists(audio_path):
            return audio_path

        raise FileNotFoundError(f"Audio file not found: {filename}")

    def _extract_sample_metadata(self, row: pd.Series) -> Dict[str, object]:
        return {
            "client_id": row.get("client_id", ""),
            "age": row.get("age", ""),
            "gender": row.get("gender", ""),
            "accents": row.get("accents", ""),
            "up_votes": row.get("up_votes", 0),
            "down_votes": row.get("down_votes", 0),
        }

    # Public API ----------------------------------------------------------
    def load_split(self, lang_code: str, split: Optional[str] = None) -> pd.DataFrame:
        target_split = split or self.default_split
        tsv_path = self._get_split_tsv_path(lang_code, target_split)
        return self._load_dataframe(tsv_path)

    def get_samples(
        self,
        lang_code: str,
        n_samples: int = 100,
        split: Optional[str] = None,
        min_up_votes: int = 2,
        require_gender: bool = False,
        require_age: bool = False,
    ) -> pd.DataFrame:
        df = self.load_split(lang_code, split)
        df = self._apply_quality_filters(df, min_up_votes, require_gender, require_age)
        df = self._sample_deterministically(df, n_samples)

        if len(df) < n_samples:
            print(f"Warning: {lang_code}/{split or self.default_split} has only {len(df)}/{n_samples} samples")

        return df.reset_index(drop=True)

    def get_audio_file_path(self, lang_code: str, audio_filename: str) -> str:
        return self._build_audio_path(lang_code, audio_filename)

    def get_sample_with_audio(self, lang_code: str, sample_row: pd.Series) -> Dict[str, object]:
        metadata = self._extract_sample_metadata(sample_row)

        return {
            "audio_path": self._build_audio_path(lang_code, sample_row["path"]),
            "text": sample_row["sentence"],
            "lang_code": lang_code,
            **metadata,
        }

    def iter_language_samples(
        self,
        n_samples: int = 100,
        split: Optional[str] = None,
        languages: Optional[Iterable[str]] = None,
        min_up_votes: int = 2,
        skip_errors: bool = True,
    ) -> Generator[tuple[str, pd.DataFrame], None, None]:
        langs_to_process = languages if languages else sorted(self.languages)

        for lang_code in langs_to_process:
            try:
                samples = self.get_samples(lang_code, n_samples, split, min_up_votes=min_up_votes)

                if len(samples) > 0:
                    yield lang_code, samples

            except Exception as exc:  # noqa: BLE001
                print(f"Error processing {lang_code}: {exc}")
                if not skip_errors:
                    raise

    def get_batch_generator(
        self,
        lang_code: str,
        samples_df: pd.DataFrame,
        batch_size: int = 16,
    ) -> Generator[List[Dict[str, object]], None, None]:
        for i in range(0, len(samples_df), batch_size):
            batch: List[Dict[str, object]] = []

            for _, row in samples_df.iloc[i : i + batch_size].iterrows():
                try:
                    sample = self.get_sample_with_audio(lang_code, row)
                    batch.append(sample)
                except Exception as exc:  # noqa: BLE001
                    print(f"Skipping sample: {exc}")
                    continue

            if batch:
                yield batch

    def get_language_stats(self, lang_code: str, split: Optional[str] = None) -> Dict[str, object]:
        try:
            df = self.load_split(lang_code, split)

            stats: Dict[str, object] = {
                "total_samples": len(df),
                "available_splits": self._get_available_splits(lang_code),
            }

            if "client_id" in df.columns:
                stats["unique_speakers"] = df["client_id"].nunique()

            if "up_votes" in df.columns:
                stats["avg_up_votes"] = float(df["up_votes"].mean())
                stats["high_quality_samples"] = int(len(df[df["up_votes"] >= 2]))

            return stats

        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def iter_language_samples_with_checkpoint(
        self,
        n_samples: int = 100,
        split: Optional[str] = None,
        languages: Optional[Iterable[str]] = None,
        checkpoint_file: str = "wer_checkpoint.json",
    ) -> Generator[tuple[str, Optional[pd.DataFrame]], None, None]:
        completed = set()
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as fh:
                checkpoint = json.load(fh)
                completed = set(checkpoint.get("completed", []))
                print(f"Resuming from checkpoint: {len(completed)} languages already done")

        langs_to_process = languages if languages else sorted(self.languages)
        langs_to_process = [lang for lang in langs_to_process if lang not in completed]

        for lang_code in langs_to_process:
            try:
                samples = self.get_samples(lang_code, n_samples, split)
                if len(samples) > 0:
                    yield lang_code, samples
            except Exception as exc:  # noqa: BLE001
                print(f"[{datetime.now()}] Error with {lang_code}: {exc}")
                yield lang_code, None


__all__ = ["CommonVoiceDataset"]
