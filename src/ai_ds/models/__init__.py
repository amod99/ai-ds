"""Small, explicit model registry for V1 tabular experiments."""

from ai_ds.models.registry import build_model, supported_models
from ai_ds.models.trainer import build_pipeline

__all__ = ["build_model", "build_pipeline", "supported_models"]
