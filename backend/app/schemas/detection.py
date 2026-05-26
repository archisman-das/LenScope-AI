"""
Detection schemas for image/video detection requests and responses
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float = Field(..., ge=0, description="Left x coordinate")
    y1: float = Field(..., ge=0, description="Top y coordinate")
    x2: float = Field(..., ge=0, description="Right x coordinate")
    y2: float = Field(..., ge=0, description="Bottom y coordinate")
    
    @validator('x2')
    def validate_x2(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @validator('y2')
    def validate_y2(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height


class DetectionResultResponse(BaseModel):
    """Individual detection result for a detected object"""
    id: Optional[int] = None
    class_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float = Field(..., ge=0, le=1)
    track_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class DetectionCreate(BaseModel):
    """Schema for creating a new detection request"""
    model_name: str = Field(..., description="Model identifier (e.g., yolov8n, yolov8s)")
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class DetectionResponse(BaseModel):
    """Full detection response with all details"""
    id: int
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    model_name: str
    confidence_threshold: float
    iou_threshold: float
    input_type: str
    input_filename: Optional[str] = None
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    num_detections: int
    inference_time_ms: Optional[float] = None
    fps: Optional[float] = None
    device_used: Optional[str] = None
    output_image_base64: Optional[str] = None
    output_path: Optional[str] = None
    auto_selection: Optional[Dict[str, Any]] = None
    use_onnx: bool
    use_fp16: bool
    created_at: datetime
    results: List[DetectionResultResponse] = []
    
    class Config:
        from_attributes = True


class DetectionListResponse(BaseModel):
    """Paginated list of detections"""
    items: List[DetectionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    class Config:
        from_attributes = True


class DetectionSummary(BaseModel):
    """Summary statistics for detections"""
    total_detections: int
    total_objects: int
    avg_confidence: float
    avg_inference_time_ms: float
    avg_fps: float
    class_distribution: Dict[str, int]
    device_distribution: Dict[str, int]
    model_distribution: Dict[str, int]
    detections_by_date: Dict[str, int]


class DetectionFilter(BaseModel):
    """Filter parameters for detection queries"""
    model_name: Optional[str] = None
    input_type: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0, le=1)
    max_confidence: Optional[float] = Field(None, ge=0, le=1)
    device_used: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    class_name: Optional[str] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class DetectionCompareRequest(BaseModel):
    """Request for comparing multiple models"""
    models: List[str] = Field(..., min_items=2, max_items=5)
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)


class DetectionCompareResponse(BaseModel):
    """Response for model comparison"""
    results: List[Dict[str, Any]]
    comparison: Dict[str, Any]
    recommendation: str


class WebcamDetectionRequest(BaseModel):
    """Request for webcam detection settings"""
    model_name: str
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    camera_index: int = Field(0, ge=0)
    width: int = Field(640, ge=160, le=1920)
    height: int = Field(480, ge=120, le=1080)
    fps: int = Field(30, ge=1, le=60)
    show_labels: bool = True
    show_confidence: bool = True
    track_objects: bool = False


class ObjectCountResponse(BaseModel):
    """Response for object counting"""
    total_objects: int
    class_counts: Dict[str, int]
    timestamp: datetime
