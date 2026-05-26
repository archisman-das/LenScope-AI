"""
Common schemas used across the application
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generic, TypeVar


# Type variable for generic paginated response
T = TypeVar('T')


class Message(BaseModel):
    """Simple message response"""
    message: str
    success: bool = True


class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str
    error_code: Optional[str] = None
    status_code: int
    timestamp: Optional[str] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response"""
    items: List[T]
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")
    
    @classmethod
    def create(cls, items: List[Any], total: int, page: int, page_size: int):
        """Factory method to create paginated response"""
        total_pages = (total + page_size - 1) // page_size
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    version: str
    database: bool = True
    models_loaded: bool = True
    cuda_available: bool = False
    timestamp: str


class ModelInfo(BaseModel):
    """Model information schema"""
    id: str
    name: str
    type: str
    description: str
    input_size: int
    recommended_for: List[str]
    weights_url: Optional[str] = None


class SystemInfo(BaseModel):
    """System information schema"""
    platform: str
    architecture: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_info: Optional[Dict[str, Any]] = None
    cpu_info: Optional[str] = None
    memory_info: Optional[Dict[str, float]] = None


class UploadProgress(BaseModel):
    """Upload progress tracking"""
    filename: str
    uploaded: int
    total: int
    progress: float  # 0-100
    status: str  # 'uploading', 'processing', 'complete', 'error'


class WebSocketMessage(BaseModel):
    """WebSocket message schema"""
    type: str  # 'detection', 'status', 'error', 'heartbeat'
    data: Dict[str, Any]
    timestamp: str


class DetectionWebSocketMessage(WebSocketMessage):
    """Detection result via WebSocket"""
    type: str = "detection"
    
    class Config:
        schema_extra = {
            "example": {
                "type": "detection",
                "data": {
                    "frame_id": 1,
                    "detections": [
                        {
                            "class": "person",
                            "confidence": 0.95,
                            "bbox": [100, 50, 200, 300]
                        }
                    ],
                    "fps": 30.5,
                    "inference_time_ms": 32.5
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class StatusWebSocketMessage(WebSocketMessage):
    """Status update via WebSocket"""
    type: str = "status"
    
    class Config:
        schema_extra = {
            "example": {
                "type": "status",
                "data": {
                    "connected": True,
                    "model": "yolov8n",
                    "device": "cuda",
                    "fps": 30
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class JobStatus(BaseModel):
    """Job status for async operations"""
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float  # 0-100
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


class BatchDetectionRequest(BaseModel):
    """Request for batch detection"""
    model_name: str
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    files: List[str]  # List of file paths or URLs


class BatchDetectionResponse(BaseModel):
    """Response for batch detection"""
    job_id: str
    total_files: int
    processed_files: int
    failed_files: int
    results: List[Any] = []
    status: str