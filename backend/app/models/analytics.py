"""
Analytics models for tracking system performance and usage patterns
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any

from ..database import Base


class AnalyticsEvent(Base):
    """
    Analytics event tracking for user behavior and system usage
    """
    
    __tablename__ = "analytics_events"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(String(100), index=True, nullable=True)
    
    # Event information
    event_type = Column(String(50), nullable=False, index=True)  # 'detection', 'model_change', 'upload', etc.
    event_category = Column(String(50), nullable=True)  # 'ai', 'ui', 'system'
    event_action = Column(String(100), nullable=False)
    event_label = Column(String(255), nullable=True)
    
    # Event data
    event_value = Column(Float, nullable=True)
    event_data = Column(JSON, nullable=True)  # Store additional event-specific data
    
    # Context
    page_url = Column(String(255), nullable=True)
    referrer = Column(String(255), nullable=True)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    
    # Device/System info
    device_type = Column(String(50), nullable=True)  # 'desktop', 'mobile', 'tablet'
    browser = Column(String(100), nullable=True)
    os = Column(String(100), nullable=True)
    screen_resolution = Column(String(20), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", back_populates="analytics_events")
    
    def __repr__(self):
        return f"<AnalyticsEvent(type='{self.event_type}', action='{self.event_action}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "event_category": self.event_category,
            "event_action": self.event_action,
            "event_label": self.event_label,
            "event_value": self.event_value,
            "event_data": self.event_data,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class ModelBenchmark(Base):
    """
    Model performance benchmarks and comparison data
    """
    
    __tablename__ = "model_benchmarks"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model information
    model_name = Column(String(50), nullable=False, index=True)
    model_type = Column(String(20), nullable=True)  # 'realtime', 'research', 'openvocabulary'
    model_version = Column(String(20), nullable=True)
    
    # Hardware information
    device_type = Column(String(20), nullable=False)  # 'cpu', 'cuda', 'tensorrt'
    device_name = Column(String(100), nullable=True)
    cpu_model = Column(String(100), nullable=True)
    gpu_model = Column(String(100), nullable=True)
    ram_gb = Column(Float, nullable=True)
    
    # Performance metrics
    avg_inference_time_ms = Column(Float, nullable=False)
    min_inference_time_ms = Column(Float, nullable=True)
    max_inference_time_ms = Column(Float, nullable=True)
    std_inference_time_ms = Column(Float, nullable=True)
    
    avg_fps = Column(Float, nullable=False)
    max_fps = Column(Float, nullable=True)
    
    # Accuracy metrics (if available)
    map_50 = Column(Float, nullable=True)  # mAP @ 0.5 IoU
    map_50_95 = Column(Float, nullable=True)  # mAP @ 0.5:0.95 IoU
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Resource usage
    avg_memory_mb = Column(Float, nullable=True)
    max_memory_mb = Column(Float, nullable=True)
    avg_cpu_usage = Column(Float, nullable=True)  # Percentage
    avg_gpu_usage = Column(Float, nullable=True)  # Percentage
    
    # Test configuration
    batch_size = Column(Integer, default=1)
    input_resolution = Column(String(20), nullable=True)  # e.g., "640x640"
    num_test_images = Column(Integer, nullable=True)
    confidence_threshold = Column(Float, default=0.5)
    
    # Optimization settings
    use_onnx = Column(Boolean, default=False)
    use_fp16 = Column(Boolean, default=False)
    use_int8 = Column(Boolean, default=False)
    
    # Metadata
    dataset_name = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<ModelBenchmark(model='{self.model_name}', fps={self.avg_fps:.1f})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark to dictionary"""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "performance": {
                "avg_inference_time_ms": self.avg_inference_time_ms,
                "min_inference_time_ms": self.min_inference_time_ms,
                "max_inference_time_ms": self.max_inference_time_ms,
                "avg_fps": self.avg_fps,
                "max_fps": self.max_fps
            },
            "accuracy": {
                "map_50": self.map_50,
                "map_50_95": self.map_50_95,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score
            },
            "resource_usage": {
                "avg_memory_mb": self.avg_memory_mb,
                "max_memory_mb": self.max_memory_mb,
                "avg_cpu_usage": self.avg_cpu_usage,
                "avg_gpu_usage": self.avg_gpu_usage
            },
            "configuration": {
                "batch_size": self.batch_size,
                "input_resolution": self.input_resolution,
                "use_onnx": self.use_onnx,
                "use_fp16": self.use_fp16,
                "use_int8": self.use_int8
            },
            "created_at": self.created_at.isoformat() if self.created_at else None
        }