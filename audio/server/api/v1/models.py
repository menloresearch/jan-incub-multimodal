from fastapi import APIRouter
from ..config import get_enabled_models, ModelInfo
from ..dependencies import WhisperModelManagerDependency
from typing import List, Dict

router = APIRouter(tags=["Models"])


@router.get(
    "/v1/models",
    summary="List available models",
    description="Returns a list of available transcription models with details about each model.",
    response_description="List of available models with their capabilities and sizes",
)
def list_models() -> List[ModelInfo]:
    """
    Get all available transcription models.

    Returns detailed information about each model including:
    - Model ID for API calls
    - Framework (faster-whisper, etc.)
    - Description of capabilities
    - Model size
    """
    return get_enabled_models()


@router.get(
    "/debug/models/status",
    summary="Model loading status",
    description="Shows which models are currently loaded in memory for debugging.",
)
def model_status(model_manager: WhisperModelManagerDependency) -> Dict:
    """Debug endpoint showing loaded models status."""
    return {
        "loaded_models": list(model_manager.loaded_models.keys()),
        "total_loaded": len(model_manager.loaded_models),
    }
