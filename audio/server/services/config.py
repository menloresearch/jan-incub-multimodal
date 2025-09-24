import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ModelFramework(str, Enum):
    FASTER_WHISPER = "faster-whisper"


@dataclass
class ModelInfo:
    id: str
    framework: ModelFramework
    description: str
    size: str = ""


# Available models by framework
AVAILABLE_MODELS: Dict[ModelFramework, List[ModelInfo]] = {
    ModelFramework.FASTER_WHISPER: [
        ModelInfo(
            "tiny", ModelFramework.FASTER_WHISPER, "Fastest, least accurate", "39 MB"
        ),
        ModelInfo(
            "tiny.en", ModelFramework.FASTER_WHISPER, "English-only, fastest", "39 MB"
        ),
        ModelInfo(
            "base", ModelFramework.FASTER_WHISPER, "Balanced speed/accuracy", "74 MB"
        ),
        ModelInfo(
            "base.en", ModelFramework.FASTER_WHISPER, "English-only, balanced", "74 MB"
        ),
        ModelInfo("small", ModelFramework.FASTER_WHISPER, "Good accuracy", "244 MB"),
        ModelInfo(
            "small.en", ModelFramework.FASTER_WHISPER, "English-only, good", "244 MB"
        ),
        ModelInfo("medium", ModelFramework.FASTER_WHISPER, "Better accuracy", "769 MB"),
        ModelInfo(
            "medium.en", ModelFramework.FASTER_WHISPER, "English-only, better", "769 MB"
        ),
        ModelInfo(
            "large-v1", ModelFramework.FASTER_WHISPER, "High accuracy v1", "1550 MB"
        ),
        ModelInfo(
            "large-v2", ModelFramework.FASTER_WHISPER, "High accuracy v2", "1550 MB"
        ),
        ModelInfo(
            "large-v3", ModelFramework.FASTER_WHISPER, "High accuracy v3", "1550 MB"
        ),
        ModelInfo(
            "large", ModelFramework.FASTER_WHISPER, "Latest large model", "1550 MB"
        ),
        ModelInfo("turbo", ModelFramework.FASTER_WHISPER, "Fast large model", "809 MB"),
    ]
}


@dataclass
class FeatureFlags:
    disabled_models: List[str] = field(
        default_factory=lambda: os.getenv("DISABLED_MODELS", "").split(",")
        if os.getenv("DISABLED_MODELS")
        else []
    )
    enabled_frameworks: List[ModelFramework] = field(
        default_factory=lambda: [ModelFramework.FASTER_WHISPER]
    )


@dataclass
class Config:
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    device: str = "cuda"
    compute_type: str = "int8"


def get_enabled_models() -> List[ModelInfo]:
    """Get all enabled models across all frameworks (all models enabled by default)"""
    config = get_config()
    enabled_models = []

    for framework in config.feature_flags.enabled_frameworks:
        if framework in AVAILABLE_MODELS:
            for model in AVAILABLE_MODELS[framework]:
                if model.id not in config.feature_flags.disabled_models:
                    enabled_models.append(model)

    return enabled_models


def get_config() -> Config:
    return Config()
