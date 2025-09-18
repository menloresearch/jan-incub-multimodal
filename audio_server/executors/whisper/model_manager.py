from faster_whisper import WhisperModel
import threading
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class WhisperModelManager:
    def __init__(self, device: str = "cuda", compute_type: str = "int8"):
        self.device = device
        self.compute_type = compute_type
        self.loaded_models: OrderedDict[str, WhisperModel] = OrderedDict()
        self._lock = threading.Lock()

    def _load_model(self, model_id: str) -> WhisperModel:
        logger.info(f"Loading model {model_id}")
        return WhisperModel(
            model_id,
            device=self.device,
            compute_type=self.compute_type,
        )

    def load_model(self, model_id: str) -> WhisperModel:
        with self._lock:
            if model_id in self.loaded_models:
                return self.loaded_models[model_id]

            model = self._load_model(model_id)
            self.loaded_models[model_id] = model
            return model
