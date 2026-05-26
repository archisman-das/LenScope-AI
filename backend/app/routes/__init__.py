"""
API Routes package
"""

from .auth import router as auth_router
from .detection import router as detection_router
from .models import router as models_router
from .analytics import router as analytics_router
from .admin import router as admin_router
from .ai_features import router as ai_features_router

__all__ = [
    "auth_router",
    "detection_router",
    "models_router",
    "analytics_router",
    "admin_router",
    "ai_features_router"
]
