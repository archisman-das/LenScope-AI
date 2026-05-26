"""
Analytics API routes for system monitoring and statistics
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, and_
from datetime import datetime, timedelta
from typing import Optional

from ..database import get_db
from ..models.detection import Detection, DetectionResult
from ..models.analytics import AnalyticsEvent, ModelBenchmark
from ..schemas.analytics import SystemStats, AnalyticsEventCreate

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/system-stats", response_model=SystemStats)
async def get_system_stats():
    """Get comprehensive system statistics"""
    import torch
    import platform
    import psutil
    
    # Get system info
    stats = SystemStats(
        platform=platform.system(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        ram_total_gb=psutil.virtual_memory().total / (1024**3),
        ram_available_gb=psutil.virtual_memory().available / (1024**3),
        total_detections=0,
        uptime_seconds=0
    )
    
    return stats


@router.get("/detection-summary")
async def get_detection_summary(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to analyze")
):
    """Get summary of detection activity over the specified period"""
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    summary = {
        "period_days": days,
        "total_detections": 0,
        "total_objects_detected": 0,
        "avg_objects_per_detection": 0,
        "avg_inference_time_ms": 0,
        "avg_fps": 0,
        "model_usage": {},
        "class_distribution": {},
        "daily_counts": []
    }
    
    # These queries would work with a real database connection
    # For now, return empty summary structure
    
    return summary


@router.get("/model-performance")
async def get_model_performance(
    model_name: Optional[str] = Query(None, description="Filter by specific model")
):
    """Get performance metrics for models"""
    performance = {
        "models": []
    }
    
    # Would query database for actual performance data
    
    return performance


@router.get("/usage-timeline")
async def get_usage_timeline(
    days: int = Query(default=30, ge=1, le=365, description="Number of days")
):
    """Get usage timeline data for charts"""
    timeline = {
        "labels": [],  # Dates
        "datasets": {
            "detections": [],
            "avg_confidence": [],
            "avg_inference_time": []
        }
    }
    
    # Generate sample data structure
    for i in range(days):
        date = (datetime.utcnow() - timedelta(days=days-i)).strftime("%Y-%m-%d")
        timeline["labels"].append(date)
        timeline["datasets"]["detections"].append(0)
        timeline["datasets"]["avg_confidence"].append(0)
        timeline["datasets"]["avg_inference_time"].append(0)
    
    return timeline


@router.get("/class-distribution")
async def get_class_distribution(
    days: int = Query(default=7, ge=1, le=90, description="Number of days")
):
    """Get distribution of detected object classes"""
    distribution = {
        "labels": [],
        "counts": [],
        "percentages": []
    }
    
    return distribution


@router.get("/popular-classes")
async def get_popular_classes(
    limit: int = Query(default=10, ge=1, le=80, description="Number of top classes")
):
    """Get most frequently detected object classes"""
    popular = []
    return {"classes": popular, "period_days": limit}


@router.post("/track-event")
async def track_analytics_event(
    event: AnalyticsEventCreate,
    db: AsyncSession = Depends(get_db)
):
    """Track an analytics event"""
    # Create event record
    db_event = AnalyticsEvent(
        event_type=event.event_type,
        event_category=event.event_category,
        event_action=event.event_action,
        event_label=event.event_label,
        event_value=event.event_value,
        event_data=event.event_data
    )
    
    db.add(db_event)
    await db.commit()
    
    return {"status": "ok", "message": "Event tracked"}