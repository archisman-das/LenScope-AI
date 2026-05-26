"""
Services package for AI/ML and business logic
"""

from .detection import DetectionService, DetectionEngine
from .model_manager import ModelManager

__all__ = [
    "DetectionService",
    "DetectionEngine",
    "ModelManager"
]