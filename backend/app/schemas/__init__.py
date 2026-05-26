"""
Pydantic schemas for request/response validation
"""

from .auth import UserCreate, UserLogin, UserResponse, Token, TokenData
from .detection import (
    DetectionCreate,
    DetectionResponse,
    DetectionResultResponse,
    DetectionListResponse,
    BoundingBox,
    DetectionSummary
)
from .analytics import (
    AnalyticsEventCreate,
    AnalyticsEventResponse,
    ModelBenchmarkCreate,
    ModelBenchmarkResponse,
    SystemStats
)
from .common import Message, PaginatedResponse

__all__ = [
    # Auth
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "Token",
    "TokenData",
    # Detection
    "DetectionCreate",
    "DetectionResponse",
    "DetectionResultResponse",
    "DetectionListResponse",
    "BoundingBox",
    "DetectionSummary",
    # Analytics
    "AnalyticsEventCreate",
    "AnalyticsEventResponse",
    "ModelBenchmarkCreate",
    "ModelBenchmarkResponse",
    "SystemStats",
    # Common
    "Message",
    "PaginatedResponse"
]