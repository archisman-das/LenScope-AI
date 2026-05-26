"""
Analytics schemas for system monitoring and performance tracking
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class AnalyticsEventCreate(BaseModel):
    """Schema for creating analytics events"""
    event_type: str = Field(..., description="Type of event (detection, upload, etc.)")
    event_category: Optional[str] = None
    event_action: str = Field(..., description="Specific action performed")
    event_label: Optional[str] = None
    event_value: Optional[float] = None
    event_data: Optional[Dict[str, Any]] = None


class AnalyticsEventResponse(BaseModel):
    """Response schema for analytics events"""
    id: int
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    event_type: str
    event_category: Optional[str] = None
    event_action: str
    event_label: Optional[str] = None
    event_value: Optional[float] = None
    event_data: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class ModelBenchmarkCreate(BaseModel):
    """Schema for creating model benchmarks"""
    model_name: str
    model_type: Optional[str] = None
    device_type: str = Field(..., description="Device type (cpu, cuda, tensorrt)")
    device_name: Optional[str] = None
    
    # Performance metrics
    avg_inference_time_ms: float
    min_inference_time_ms: Optional[float] = None
    max_inference_time_ms: Optional[float] = None
    std_inference_time_ms: Optional[float] = None
    avg_fps: float
    max_fps: Optional[float] = None
    
    # Accuracy metrics
    map_50: Optional[float] = None
    map_50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Resource usage
    avg_memory_mb: Optional[float] = None
    max_memory_mb: Optional[float] = None
    avg_cpu_usage: Optional[float] = None
    avg_gpu_usage: Optional[float] = None
    
    # Configuration
    batch_size: int = 1
    input_resolution: Optional[str] = None
    use_onnx: bool = False
    use_fp16: bool = False
    use_int8: bool = False


class ModelBenchmarkResponse(BaseModel):
    """Response schema for model benchmarks"""
    id: int
    model_name: str
    model_type: Optional[str] = None
    device_type: str
    device_name: Optional[str] = None
    performance: Dict[str, Optional[float]]
    accuracy: Dict[str, Optional[float]]
    resource_usage: Dict[str, Optional[float]]
    configuration: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class SystemStats(BaseModel):
    """System statistics response"""
    # System info
    platform: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    gpu_count: int = 0
    gpu_names: List[str] = []
    cpu_model: Optional[str] = None
    ram_total_gb: Optional[float] = None
    ram_available_gb: Optional[float] = None
    
    # Model info
    loaded_models: List[str] = []
    default_model: str = ""
    
    # Performance stats
    total_detections: int = 0
    avg_inference_time_ms: Optional[float] = None
    avg_fps: Optional[float] = None
    
    # Uptime
    uptime_seconds: float


class PerformanceMetrics(BaseModel):
    """Performance metrics for a specific time period"""
    period: str  # '1h', '24h', '7d', '30d'
    total_requests: int
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float
    success_rate: float


class UsageStatistics(BaseModel):
    """Usage statistics"""
    period: str
    total_users: int
    active_users: int
    total_detections: int
    total_images_processed: int
    total_video_frames_processed: int
    avg_detections_per_user: float
    most_used_model: str
    most_detected_class: str


class ChartData(BaseModel):
    """Generic chart data structure"""
    labels: List[str]
    datasets: List[Dict[str, Any]]


class DetectionTimeline(BaseModel):
    """Detection timeline data for charts"""
    timestamps: List[datetime]
    counts: List[int]
    avg_confidence: List[float]
    avg_inference_time: List[float]


class ClassDistribution(BaseModel):
    """Class distribution data for charts"""
    labels: List[str]
    counts: List[int]
    percentages: List[float]


class ModelComparisonData(BaseModel):
    """Data for model comparison charts"""
    models: List[str]
    metrics: Dict[str, List[float]]  # metric_name -> [values for each model]
    recommendations: List[str]