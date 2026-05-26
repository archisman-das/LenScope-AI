"""
AI Features API routes for advanced AI capabilities
- Adaptive AI Model Switching
- Scene Memory Timeline
- Object Relationship Graph
- Prediction Engine
- AI Decision Explanation Panel
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from typing import Optional, List
from datetime import datetime
import logging
from pathlib import Path

from ..services.adaptive_ai import get_adaptive_switcher, get_recommendation_engine, HardwareResourceDetector
from ..utils.image import load_image
from ..services.scene_memory import get_scene_memory
from ..services.object_relationships import get_object_graph
from ..services.prediction_engine import get_prediction_engine
from ..services.explanation_engine import get_explanation_engine
from ..services.gradcam_service import gradcam_service
from ..schemas.common import Message
from ..config import settings, AVAILABLE_MODELS
from ..utils.image import validate_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Features"])


# ============================================================================
# Adaptive AI Model Switching Routes
# ============================================================================

@router.get("/adaptive/status")
async def get_adaptive_status():
    """Get current adaptive AI system status"""
    switcher = get_adaptive_switcher()
    
    return {
        "current_model": switcher.current_model,
        "last_switch_time": switcher.last_switch_time.isoformat() if switcher.last_switch_time else None,
        "switch_cooldown": switcher.switch_cooldown,
        "total_switches": len(switcher.model_switches),
        "scene_analyses": len(switcher.scene_history)
    }


@router.get("/adaptive/switch-history")
async def get_switch_history(
    limit: int = Query(default=50, ge=1, le=100)
):
    """Get recent model switch history"""
    switcher = get_adaptive_switcher()
    return {
        "switches": switcher.get_switch_history(limit),
        "total": len(switcher.model_switches)
    }


@router.get("/adaptive/performance-stats")
async def get_model_performance_stats():
    """Get performance statistics for all models"""
    switcher = get_adaptive_switcher()
    return {
        "model_performance": switcher.get_model_performance_stats()
    }


@router.get("/adaptive/explain-routing")
async def explain_adaptive_routing(
    image_path: Optional[str] = Query(
        default=None,
        description="Optional local image path. If omitted, returns the latest routing explanation."
    )
):
    """Explain the trained meta-router's latest or image-specific model choice."""
    router = get_adaptive_switcher()

    try:
        image = None
        if image_path:
            path = Path(image_path)
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
            image = load_image(path)

        explanation = router.explain(image)
        return {
            "predicted_model": explanation["predicted_model"],
            "confidence": explanation["confidence"],
            "top3_alternatives": explanation.get("top3_alternatives", []),
            "scene_features_used": explanation.get("scene_features_used"),
            "feature_vector": explanation.get("feature_vector"),
            "feature_importances": explanation.get("feature_importances"),
            "gradcam_attribution": explanation.get("gradcam_attribution"),
            "router_kind": explanation.get("router_kind"),
            "router_model_path": explanation.get("router_model_path"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to explain adaptive routing: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to explain routing: {str(e)}")


@router.post("/adaptive/reset")
async def reset_adaptive_statistics():
    """Reset adaptive AI statistics"""
    switcher = get_adaptive_switcher()
    switcher.reset_statistics()
    return Message(message="Adaptive AI statistics reset successfully")


# ============================================================================
# Scene Memory Timeline Routes
# ============================================================================

@router.get("/scene/timeline")
async def get_scene_timeline(
    limit: int = Query(default=50, ge=1, le=100),
    session_id: Optional[str] = None,
    class_filter: Optional[str] = None,
    start_time: Optional[str] = Query(default=None, description="ISO format start time"),
    end_time: Optional[str] = Query(default=None, description="ISO format end time"),
    search: Optional[str] = Query(default=None, description="Search in detections")
):
    """
    Get scene timeline with optional filtering and search
    
    Returns timeline cards with visual events, detection history, and metadata.
    Supports filtering by class, time range, and text search.
    """
    memory = get_scene_memory()
    
    # Get base timeline
    from datetime import datetime
    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None
    
    timeline = memory.get_timeline(limit, session_id, start_dt, end_dt)
    
    # Apply class filter
    if class_filter:
        filtered_timeline = []
        for frame in timeline:
            filtered_detections = [d for d in frame["detections"] if d["class_name"] == class_filter]
            if filtered_detections:
                frame["detections"] = filtered_detections
                frame["object_count"] = len(filtered_detections)
                filtered_timeline.append(frame)
        timeline = filtered_timeline
    
    # Apply text search
    if search:
        search_lower = search.lower()
        filtered_timeline = []
        for frame in timeline:
            # Search in class names and metadata
            matching_detections = [
                d for d in frame["detections"] 
                if search_lower in d.get("class_name", "").lower() or
                search_lower in str(frame.get("metadata", {})).lower()
            ]
            if matching_detections or search_lower in str(frame.get("metadata", {})).lower():
                if matching_detections:
                    frame["detections"] = matching_detections
                    frame["object_count"] = len(matching_detections)
                filtered_timeline.append(frame)
        timeline = filtered_timeline
    
    return {
        "timeline": timeline,
        "total_frames": memory.scene_statistics["total_frames"],
        "filtered_count": len(timeline)
    }


@router.get("/scene/events")
async def get_scene_events(
    event_type: Optional[str] = Query(default=None, description="Filter by event type: object_entered, object_exited, crowd_formed, etc."),
    limit: int = Query(default=50, ge=1, le=100)
):
    """
    Get important visual events from scene memory
    
    Events include:
    - object_entered: New object appeared in scene
    - object_exited: Object disappeared from scene
    - crowd_formed: Multiple objects gathered
    - scene_change: Significant scene composition change
    - tracking_lost: Object tracking was interrupted
    """
    memory = get_scene_memory()
    events = []
    
    # Generate events from temporal patterns and tracks
    for pattern in memory.temporal_patterns[-limit:]:
        if pattern["object_count_trend"] == "increasing":
            events.append({
                "event_type": "objects_entering",
                "timestamp": pattern["timestamp"],
                "description": f"Object count increasing (avg {pattern['avg_objects_per_frame']} per frame)",
                "frame_range": pattern["frame_range"],
                "metadata": pattern
            })
        elif pattern["object_count_trend"] == "decreasing":
            events.append({
                "event_type": "objects_leaving",
                "timestamp": pattern["timestamp"],
                "description": f"Object count decreasing (avg {pattern['avg_objects_per_frame']} per frame)",
                "frame_range": pattern["frame_range"],
                "metadata": pattern
            })
    
    # Generate events from tracks
    for track in memory.object_tracks.values():
        if len(track.observations) == 1:
            events.append({
                "event_type": "object_entered",
                "timestamp": track.first_seen.isoformat(),
                "description": f"New {track.class_name} detected",
                "track_id": track.track_id,
                "metadata": {"class_name": track.class_name}
            })
    
    # Filter by event type
    if event_type:
        events = [e for e in events if e["event_type"] == event_type]
    
    # Sort by timestamp (newest first) and limit
    events.sort(key=lambda e: e["timestamp"], reverse=True)
    events = events[:limit]
    
    return {
        "events": events,
        "total": len(events)
    }


@router.get("/scene/history")
async def get_detection_history(
    class_name: Optional[str] = Query(default=None, description="Filter by class name"),
    start_time: Optional[str] = Query(default=None, description="ISO format start time"),
    end_time: Optional[str] = Query(default=None, description="ISO format end time"),
    limit: int = Query(default=100, ge=1, le=500)
):
    """
    Get detailed detection history
    
    Returns a chronological history of all detections with filtering options.
    """
    memory = get_scene_memory()
    
    from datetime import datetime
    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None
    
    history = []
    
    for frame in list(memory.frames):
        # Apply time filters
        if start_dt and frame.timestamp < start_dt:
            continue
        if end_dt and frame.timestamp > end_dt:
            continue
        
        # Get detections for this frame
        detections = frame.detections
        if class_name:
            detections = [d for d in detections if d["class_name"] == class_name]
        
        if detections:
            history.append({
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp.isoformat(),
                "model_used": frame.model_used,
                "session_id": frame.session_id,
                "detections": detections,
                "total_objects": len(detections)
            })
        
        if len(history) >= limit:
            break
    
    return {
        "history": history,
        "total_entries": len(history)
    }


@router.get("/scene/statistics")
async def get_scene_statistics():
    """Get scene statistics"""
    memory = get_scene_memory()
    return memory.get_scene_statistics()


@router.get("/scene/tracks")
async def get_object_tracks(
    class_filter: Optional[str] = None,
    min_observations: int = Query(default=1, ge=1),
    limit: int = Query(default=100, ge=1, le=500)
):
    """Get object tracks"""
    memory = get_scene_memory()
    return {
        "tracks": memory.get_object_tracks(class_filter, min_observations, limit),
        "total_tracks": len(memory.object_tracks)
    }


@router.get("/scene/track/{track_id}")
async def get_track_trajectory(track_id: str):
    """Get detailed trajectory for a specific track"""
    memory = get_scene_memory()
    trajectory = memory.get_track_trajectory(track_id)
    
    if trajectory is None:
        raise HTTPException(status_code=404, detail=f"Track {track_id} not found")
    
    return trajectory


@router.get("/scene/session/{session_id}")
async def get_session_summary(session_id: str):
    """Get summary for a specific session"""
    memory = get_scene_memory()
    summary = memory.get_session_summary(session_id)
    
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return summary


@router.get("/scene/class-timeline/{class_name}")
async def get_class_timeline(
    class_name: str,
    limit: int = Query(default=100, ge=1, le=500)
):
    """Get timeline of detections for a specific class"""
    memory = get_scene_memory()
    return {
        "class_name": class_name,
        "timeline": memory.get_class_timeline(class_name, limit)
    }


@router.delete("/scene/clear")
async def clear_scene_memory():
    """Clear scene memory"""
    memory = get_scene_memory()
    memory.clear()
    return Message(message="Scene memory cleared successfully")


# ============================================================================
# Object Relationship Graph Routes
# ============================================================================

@router.get("/graph/overview")
async def get_graph_overview():
    """Get overview of object relationship graph"""
    graph = get_object_graph()
    return graph.get_relationship_statistics()


@router.get("/graph/full")
async def get_full_graph(
    include_edges: bool = Query(default=True, description="Include edge details")
):
    """Get full graph structure"""
    graph = get_object_graph()
    return graph.get_graph(include_edges)


@router.get("/graph/node/{node_id}")
async def get_node_details(node_id: str):
    """Get details for a specific node"""
    graph = get_object_graph()
    node = graph.get_node(node_id)
    
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    
    return node


@router.get("/graph/class/{class_name}/relationships")
async def get_class_relationships(class_name: str):
    """Get all relationships involving a specific class"""
    graph = get_object_graph()
    return graph.get_relationships_for_class(class_name)


@router.get("/graph/co-occurrence")
async def get_co_occurrence_matrix():
    """Get co-occurrence matrix for all classes"""
    graph = get_object_graph()
    return {
        "co_occurrence_matrix": graph.get_co_occurrence_matrix(),
        "top_pairs": graph._get_top_co_occurrences(20)
    }


@router.get("/graph/frame/{frame_id}")
async def get_frame_subgraph(frame_id: str):
    """Get subgraph for a specific frame"""
    graph = get_object_graph()
    return graph.get_subgraph_for_frame(frame_id)


@router.delete("/graph/clear")
async def clear_object_graph():
    """Clear object relationship graph"""
    graph = get_object_graph()
    graph.clear()
    return Message(message="Object relationship graph cleared successfully")


# ============================================================================
# Prediction Engine Routes
# ============================================================================

@router.get("/predictions/scene-evolution")
async def predict_scene_evolution(
    frames_ahead: int = Query(default=5, ge=1, le=30)
):
    """Predict how the scene will evolve"""
    engine = get_prediction_engine()
    return engine.predict_scene_evolution(frames_ahead)


@router.get("/predictions/next-frame")
async def predict_next_frame():
    """Get predictions for the next frame"""
    engine = get_prediction_engine()
    return engine.get_next_frame_predictions()


@router.get("/predictions/trajectory/{track_id}")
async def predict_trajectory(
    track_id: str,
    time_horizon: float = Query(default=1.0, ge=0.1, le=10.0),
    method: str = Query(default="linear", description="Prediction method: linear, polynomial")
):
    """Predict future position of a tracked object"""
    engine = get_prediction_engine()
    prediction = engine.predict_object_trajectory(track_id, time_horizon, method)
    
    if prediction is None:
        raise HTTPException(
            status_code=404, 
            detail=f"No trajectory data available for track {track_id}"
        )
    
    return prediction


@router.get("/predictions/accuracy")
async def get_prediction_accuracy():
    """Get prediction accuracy statistics"""
    engine = get_prediction_engine()
    return engine.get_prediction_accuracy_stats()


@router.get("/predictions/summary")
async def get_prediction_summary():
    """Get comprehensive prediction summary"""
    engine = get_prediction_engine()
    return engine.get_prediction_summary()


@router.delete("/predictions/clear")
async def clear_predictions():
    """Clear prediction data"""
    engine = get_prediction_engine()
    engine.clear()
    return Message(message="Prediction data cleared successfully")


# ============================================================================
# AI Decision Explanation Routes
# ============================================================================

@router.get("/explanations/history")
async def get_explanation_history(
    limit: int = Query(default=50, ge=1, le=100)
):
    """Get recent explanation history"""
    engine = get_explanation_engine()
    return {
        "explanations": engine.get_explanation_history(limit),
        "total": len(engine.explanation_history)
    }


@router.post("/explain/gradcam")
async def explain_gradcam(
    file: UploadFile = File(..., description="Image file to explain"),
    model_id: str = Form(default="yolov8n", description="Model identifier"),
    confidence_threshold: float = Form(default=0.25, ge=0, le=1),
    iou_threshold: float = Form(default=0.45, ge=0, le=1),
):
    """Generate GradCAM overlay and activation-aware detection explanations."""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    if not AVAILABLE_MODELS[model_id].get("runnable", True):
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' is not runnable")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    is_valid, error_msg, image = validate_image(content, settings.MAX_UPLOAD_SIZE)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    result = gradcam_service.explain(
        image_source=image,
        model_id=model_id,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    detections = gradcam_service.enrich_detections_with_activation(
        result["detections"],
        result["per_class_activation_map"],
    )
    explanation_engine = get_explanation_engine()
    image_dimensions = {
        "width": image.shape[1],
        "height": image.shape[0],
    }

    return {
        "model_id": model_id,
        "detection_boxes": detections,
        "gradcam_overlay_url": result["gradcam_overlay_url"],
        "per_class_activation_map": result["per_class_activation_map"],
        "explanations": [
            explanation_engine.explain_detection(detection, image_dimensions)
            for detection in detections[:10]
        ],
    }


@router.post("/explanations/detection")
async def explain_detection(
    detection: dict,
    image_width: int = Query(..., description="Image width"),
    image_height: int = Query(..., description="Image height")
):
    """Explain a single detection"""
    engine = get_explanation_engine()
    image_dimensions = {"width": image_width, "height": image_height}
    return engine.explain_detection(detection, image_dimensions)


@router.post("/explanations/model-selection")
async def explain_model_selection(
    model_id: str,
    selection_reasons: List[str],
    scene_complexity: Optional[dict] = None,
    use_case: str = Query(default="auto")
):
    """Explain model selection decision"""
    engine = get_explanation_engine()
    return engine.explain_model_selection(
        model_id, selection_reasons, scene_complexity, use_case
    )


@router.delete("/explanations/clear")
async def clear_explanations():
    """Clear explanation history"""
    engine = get_explanation_engine()
    engine.clear()
    return Message(message="Explanation history cleared successfully")


# ============================================================================
# Combined AI Analysis Route
# ============================================================================

@router.get("/analysis/full")
async def get_full_ai_analysis():
    """Get comprehensive AI analysis from all systems"""
    switcher = get_adaptive_switcher()
    memory = get_scene_memory()
    graph = get_object_graph()
    prediction = get_prediction_engine()
    explanation = get_explanation_engine()
    
    return {
        "adaptive_ai": {
            "current_model": switcher.current_model,
            "performance_stats": switcher.get_model_performance_stats(),
            "recent_switches": switcher.get_switch_history(5)
        },
        "scene_memory": {
            "statistics": memory.get_scene_statistics(),
            "recent_frames": memory.get_timeline(5)
        },
        "object_graph": {
            "statistics": graph.get_relationship_statistics(),
            "top_co_occurrences": graph._get_top_co_occurrences(10)
        },
        "predictions": {
            "next_frame": prediction.get_next_frame_predictions(),
            "accuracy_stats": prediction.get_prediction_accuracy_stats()
        },
        "explanations": {
            "recent": explanation.get_explanation_history(5)
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Model Recommendation Routes (NEW)
# ============================================================================

@router.get("/recommend/model")
async def recommend_model(
    scene_type: Optional[str] = Query(default=None, description="Scene type hint: plain_text, simple, crowded, etc."),
    priority: str = Query(default="auto", description="Priority: auto, speed, accuracy, resources")
):
    """
    Get AI model recommendation based on scene complexity and hardware resources
    
    Returns:
        - recommended_model: The best model for current conditions
        - selection_reason: Why this model was chosen
        - expected_performance: FPS, accuracy, inference time
        - scene_analysis: Detected scene characteristics
        - hardware_status: Current system resource status
        - alternatives: Other suitable models
    """
    engine = get_recommendation_engine()
    
    try:
        recommendation = engine.recommend(
            image=None,  # No image provided, use hardware-based recommendation
            scene_type=scene_type,
            priority=priority
        )
        return recommendation
    except Exception as e:
        logger.error(f"Error generating model recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendation: {str(e)}")


@router.post("/recommend/model/analyze")
async def recommend_model_with_image(
    scene_type: Optional[str] = Query(default=None, description="Scene type hint"),
    priority: str = Query(default="auto", description="Priority mode"),
    image_data: Optional[dict] = None  # Base64 encoded image or image path
):
    """
    Get AI model recommendation with image analysis
    
    Analyzes the provided image to determine scene complexity and recommends
    the most suitable model based on both scene characteristics and hardware resources.
    """
    engine = get_recommendation_engine()
    
    try:
        # If image data is provided, we would analyze it
        # For now, we'll use the scene_type hint if provided
        recommendation = engine.recommend(
            image=None,  # Image analysis would go here
            scene_type=scene_type,
            priority=priority
        )
        return recommendation
    except Exception as e:
        logger.error(f"Error analyzing image for recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {str(e)}")


@router.get("/hardware/resources")
async def get_hardware_resources():
    """
    Get current hardware resource status
    
    Returns detailed information about:
        - CPU: cores, usage
        - RAM: total, available
        - GPU: availability, memory, model
        - Resource level classification
    """
    resources = HardwareResourceDetector.get_system_resources()
    return resources


@router.get("/models/list")
async def list_available_models():
    """
    Get list of all available models with their specifications
    
    Returns model details including:
        - Name, type, description
        - Minimum hardware requirements
        - Expected performance (FPS on CPU/GPU)
        - Accuracy score
        - Best use cases
    """
    from ..config import AVAILABLE_MODELS
    
    models = []
    for model_id, config in AVAILABLE_MODELS.items():
        models.append({
            "id": model_id,
            "name": config["name"],
            "family": config.get("family", config["type"]),
            "type": config["type"],
            "backend": config.get("backend", "ultralytics"),
            "runnable": config.get("runnable", True),
            "description": config["description"],
            "input_size": config["input_size"],
            "recommended_for": config["recommended_for"],
            "min_ram_gb": config.get("min_ram_gb", 2),
            "min_gpu_memory_gb": config.get("min_gpu_memory_gb", 0),
            "expected_fps_cpu": config.get("expected_fps_cpu", 0),
            "expected_fps_gpu": config.get("expected_fps_gpu", 0),
            "accuracy_score": config.get("accuracy_score", 0.5),
            "best_for": config.get("best_for", "General purpose")
        })
    
    return {"models": models, "total": len(models)}


@router.get("/recommend/scenarios")
async def get_scenario_recommendations():
    """
    Get model recommendations for common scenarios
    
    Provides pre-computed recommendations for typical use cases:
        - Plain text / document scanning
        - Simple scene (few objects)
        - Crowded scene (many objects)
        - Low resource devices
        - High accuracy requirements
        - Real-time processing
    """
    engine = get_recommendation_engine()
    hardware = HardwareResourceDetector.get_system_resources()
    
    scenarios = {
        "plain_text": {
            "description": "Document scanning, text recognition, simple content",
            "recommended_model": None,
            "reason": None
        },
        "simple_scene": {
            "description": "Few objects, clear backgrounds, well-lit scenes",
            "recommended_model": None,
            "reason": None
        },
        "crowded_scene": {
            "description": "Many objects, overlapping items, complex backgrounds",
            "recommended_model": None,
            "reason": None
        },
        "low_resources": {
            "description": "Limited RAM (<4GB), no GPU, mobile devices",
            "recommended_model": None,
            "reason": None
        },
        "high_accuracy": {
            "description": "Maximum precision required, complex detection tasks",
            "recommended_model": None,
            "reason": None
        },
        "realtime": {
            "description": "Video processing, live detection, high FPS required",
            "recommended_model": None,
            "reason": None
        }
    }
    
    # Get recommendations for each scenario
    scenario_params = {
        "plain_text": {"scene_type": "plain_text", "priority": "speed"},
        "simple_scene": {"scene_type": "simple", "priority": "speed"},
        "crowded_scene": {"scene_type": "crowded", "priority": "accuracy"},
        "low_resources": {"scene_type": None, "priority": "resources"},
        "high_accuracy": {"scene_type": None, "priority": "accuracy"},
        "realtime": {"scene_type": None, "priority": "speed"}
    }
    
    for scenario_key, params in scenario_params.items():
        try:
            rec = engine.recommend(
                image=None,
                scene_type=params["scene_type"],
                priority=params["priority"]
            )
            scenarios[scenario_key]["recommended_model"] = rec["recommended_model"]
            scenarios[scenario_key]["reason"] = rec["selection_reason"]
            scenarios[scenario_key]["expected_performance"] = rec["expected_performance"]
        except Exception as e:
            logger.warning(f"Error getting recommendation for {scenario_key}: {e}")
    
    return {
        "scenarios": scenarios,
        "current_hardware": {
            "resource_level": hardware["resource_level"],
            "gpu_available": hardware["gpu"]["available"],
            "ram_available_gb": hardware["ram"]["available_gb"]
        }
    }
