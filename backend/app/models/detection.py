"""
Detection models for storing detection results and history
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..database import Base


class Detection(Base):
    """
    Detection session/model run record
    Stores information about each detection request
    """
    
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Null for anonymous users
    session_id = Column(String(100), index=True, nullable=True)  # For tracking without auth
    
    # Detection parameters
    model_name = Column(String(50), nullable=False)
    confidence_threshold = Column(Float, default=0.5)
    iou_threshold = Column(Float, default=0.45)
    
    # Input information
    input_type = Column(String(20), nullable=False)  # 'image', 'webcam', 'video'
    input_filename = Column(String(255), nullable=True)
    input_path = Column(String(500), nullable=True)
    original_width = Column(Integer, nullable=True)
    original_height = Column(Integer, nullable=True)
    
    # Output information
    output_path = Column(String(500), nullable=True)
    num_detections = Column(Integer, default=0)
    
    # Performance metrics
    inference_time_ms = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    device_used = Column(String(20), nullable=True)  # 'cpu', 'cuda'
    
    # Processing options
    use_onnx = Column(Boolean, default=False)
    use_fp16 = Column(Boolean, default=False)
    auto_downscale = Column(Boolean, default=False)
    
    # Metadata
    tags = Column(String(255), nullable=True)  # Comma-separated tags
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="detections")
    results = relationship("DetectionResult", back_populates="detection", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Detection(id={self.id}, model='{self.model_name}', detections={self.num_detections})>"
    
    def to_dict(self, include_results: bool = False) -> Dict[str, Any]:
        """Convert detection record to dictionary"""
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "input_type": self.input_type,
            "input_filename": self.input_filename,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "num_detections": self.num_detections,
            "inference_time_ms": self.inference_time_ms,
            "fps": self.fps,
            "device_used": self.device_used,
            "use_onnx": self.use_onnx,
            "use_fp16": self.use_fp16,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "tags": self.tags.split(",") if self.tags else []
        }
        
        if include_results:
            data["results"] = [result.to_dict() for result in self.results]
        
        return data


class DetectionResult(Base):
    """
    Individual detection result for each detected object
    """
    
    __tablename__ = "detection_results"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=False)
    
    # Object information
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False)
    
    # Bounding box coordinates (normalized 0-1 or pixel values)
    bbox_x1 = Column(Float, nullable=False)  # x1 (left)
    bbox_y1 = Column(Float, nullable=False)  # y1 (top)
    bbox_x2 = Column(Float, nullable=False)  # x2 (right)
    bbox_y2 = Column(Float, nullable=False)  # y2 (bottom)
    
    # Detection confidence
    confidence = Column(Float, nullable=False)
    
    # Tracking ID (for video/webcam)
    track_id = Column(Integer, nullable=True)
    
    # Additional metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)
    extra_data = Column(JSON, nullable=True)  # Store additional info like segmentation masks
    
    # Timestamp for real-time detections
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    detection = relationship("Detection", back_populates="results")
    
    def __repr__(self):
        return f"<DetectionResult(class='{self.class_name}', confidence={self.confidence:.3f})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to dictionary"""
        return {
            "id": self.id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": {
                "x1": self.bbox_x1,
                "y1": self.bbox_y1,
                "x2": self.bbox_x2,
                "y2": self.bbox_y2
            },
            "confidence": self.confidence,
            "track_id": self.track_id,
            "metadata": self.extra_data
        }
    
    @property
    def width(self) -> float:
        """Calculate bounding box width"""
        return self.bbox_x2 - self.bbox_x1
    
    @property
    def height(self) -> float:
        """Calculate bounding box height"""
        return self.bbox_y2 - self.bbox_y1
    
    @property
    def area(self) -> float:
        """Calculate bounding box area"""
        return self.width * self.height
    
    @property
    def center(self) -> tuple:
        """Calculate bounding box center point"""
        return (
            (self.bbox_x1 + self.bbox_x2) / 2,
            (self.bbox_y1 + self.bbox_y2) / 2
        )