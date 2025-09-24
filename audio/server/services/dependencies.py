from functools import lru_cache
from typing import Annotated
from fastapi import Depends, UploadFile
from faster_whisper.audio import decode_audio
from numpy import float32
from numpy.typing import NDArray

from .executors.whisper.model_manager import WhisperModelManager
from .config import Config, get_config


@lru_cache
def get_model_manager() -> WhisperModelManager:
    config = get_config()
    return WhisperModelManager(device=config.device, compute_type=config.compute_type)


WhisperModelManagerDependency = Annotated[
    WhisperModelManager, Depends(get_model_manager)
]
ConfigDependency = Annotated[Config, Depends(get_config)]


def audio_file_dependency(file: UploadFile) -> NDArray[float32]:
    return decode_audio(file.file)


AudioFileDependency = Annotated[NDArray[float32], Depends(audio_file_dependency)]
