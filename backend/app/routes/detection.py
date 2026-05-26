"""
Detection API routes for image and video object detection
"""

import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging
import uuid
from pathlib import Path
from datetime import datetime

from ..database import get_db
from ..schemas.detection import (
    DetectionCreate,
    DetectionResponse,
    DetectionCompareRequest,
    DetectionCompareResponse
)
from ..schemas.common import Message, HealthCheck
from ..services.detection import DetectionService
from ..services.model_manager import ModelManager
from ..config import settings, AVAILABLE_MODELS, COCO_CLASSES, OPEN_VOCAB_DEFAULT_CLASSES
from ..utils.image import validate_image, save_image, generate_detection_filename
from ..utils.system import get_system_info

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detection", tags=["Detection"])

# Detection service singleton
detection_service = DetectionService()


def parse_open_vocab_prompt(prompt: Optional[str]) -> List[str]:
    """Parse comma/newline-separated open-vocabulary labels."""
    if not prompt:
        return OPEN_VOCAB_DEFAULT_CLASSES

    labels = [
        label.strip().lower()
        for chunk in prompt.splitlines()
        for label in chunk.split(",")
        if label.strip()
    ]
    return labels or OPEN_VOCAB_DEFAULT_CLASSES


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Check API health and system status"""
    system_info = get_system_info()
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        database=True,
        models_loaded=len(detection_service.engines) > 0,
        cuda_available=system_info.get("cuda_available", False),
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/models")
async def list_models():
    """Get list of available detection models"""
    manager = ModelManager()
    models = []
    
    for model_id, config in AVAILABLE_MODELS.items():
        model_info = manager.get_model_info(model_id)
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
            "expected_fps_cpu": config.get("expected_fps_cpu", 0),
            "expected_fps_gpu": config.get("expected_fps_gpu", 0),
            "accuracy_score": config.get("accuracy_score", 0.5),
            "best_for": config.get("best_for", "General purpose"),
            "is_loaded": model_info.get("is_loaded", False) if model_info else False
        })
    
    return {
        "models": models,
        "total": len(models),
        "system": {
            "cuda_available": torch.cuda.is_available(),
            "device": str(manager.get_device()),
            "recommended_model": manager.recommend_model("realtime")
        }
    }


@router.get("/classes")
async def list_classes():
    """Get list of detectable object classes (COCO)"""
    return {
        "classes": COCO_CLASSES,
        "total": len(COCO_CLASSES),
        "dataset": "COCO"
    }


@router.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(..., description="Image file to detect objects in"),
    model_name: str = Form(default="auto", description="Model to use for detection (use 'auto' for automatic selection)"),
    confidence_threshold: float = Form(default=0.5, ge=0, le=1, description="Confidence threshold"),
    iou_threshold: float = Form(default=0.45, ge=0, le=1, description="IoU threshold for NMS"),
    draw_boxes: bool = Form(default=True, description="Whether to draw bounding boxes"),
    open_vocab_prompt: Optional[str] = Form(None, description="Optional labels for open-vocabulary detection"),
    session_id: Optional[str] = Form(None, description="Session ID for tracking")
):
    """
    Detect objects in an uploaded image
    
    - **file**: Image file (JPG, PNG, BMP, WebP, GIF, TIFF, etc. - any size supported)
    - **model_name**: Model identifier (yolov8n, yolov8s, etc.) or "auto" for automatic selection
    - **confidence_threshold**: Minimum confidence for detections (0-1)
    - **iou_threshold**: IoU threshold for NMS (0-1)
    - **draw_boxes**: Whether to draw bounding boxes on output
    
    Note: This endpoint supports images of any size and various formats.
    Detection may return empty results if no objects are detected.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file content
    content = await file.read()
    
    # Check for empty file
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    # Validate image (now supports unlimited size with -1)
    is_valid, error_msg, image = validate_image(content, settings.MAX_UPLOAD_SIZE)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    open_vocab_classes = parse_open_vocab_prompt(open_vocab_prompt)
    auto_mode = model_name == "auto"

    # Check if model is available
    if not auto_mode and model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}"
        )
    if not auto_mode and not AVAILABLE_MODELS[model_name].get("runnable", True):
        # Fallback to yolov8n if selected model is not runnable
        logger.warning(f"Model '{model_name}' is not runnable, falling back to yolov8n")
        model_name = "yolov8n"
    
    try:
        # Run detection (now with graceful degradation - returns empty list on failure)
        if auto_mode:
            result = detection_service.detect_image_best(
                image_source=image,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                draw_boxes=draw_boxes,
                return_base64=True,
                open_vocab_classes=open_vocab_classes,
            )
            model_name = result["model_id"]
            logger.info("Auto-selected model: %s for image detection", model_name)
        else:
            result = detection_service.detect_image(
                image_source=image,
                model_id=model_name,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                draw_boxes=draw_boxes,
                return_base64=True,
                open_vocab_classes=open_vocab_classes,
            )
        
        # Generate output filename
        output_filename = generate_detection_filename(file.filename)
        output_path = settings.upload_path / output_filename
        
        # Save output image if we have detections
        if draw_boxes and result.get("output_image_base64"):
            import base64
            from ..utils.image import load_image
            image_data = result["output_image_base64"].split(",", 1)[-1]
            output_image = load_image(base64.b64decode(image_data))
            save_image(output_image, str(output_path))
            result["output_path"] = str(output_path)
        
        # Create detection record in database
        from sqlalchemy.ext.asyncio import AsyncSession
        from ..models.detection import Detection, DetectionResult
        from ..utils.helpers import generate_session_id
        
        detection_id = None
        if session_id is None:
            session_id = generate_session_id()
        
        # Note: Database operations would be done here with async session
        # For now, return the result directly
        
        return DetectionResponse(
            id=detection_id or 0,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_type="image",
            input_filename=file.filename,
            original_width=result["original_width"],
            original_height=result["original_height"],
            num_detections=result["num_detections"],
            inference_time_ms=result["inference_time_ms"],
            fps=result["fps"],
            device_used=result["device_used"],
            output_image_base64=result.get("output_image_base64"),
            output_path=result.get("output_path"),
            auto_selection=result.get("auto_selection"),
            use_onnx=False,
            use_fp16=False,
            created_at=datetime.utcnow(),
            results=[
                {
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "bbox": det["bbox"],
                    "confidence": det["confidence"]
                }
                for det in result["detections"]
            ]
        )
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        # Return empty results instead of error for graceful degradation
        return DetectionResponse(
            id=0,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_type="image",
            input_filename=file.filename,
            original_width=0,
            original_height=0,
            num_detections=0,
            inference_time_ms=0,
            fps=0,
            device_used="error",
            output_image_base64=None,
            output_path=None,
            auto_selection=None,
            use_onnx=False,
            use_fp16=False,
            created_at=datetime.utcnow(),
            results=[]
        )


@router.post("/recommend/image")
async def recommend_model_for_image(
    file: UploadFile = File(..., description="Image file to analyze for model selection"),
    open_vocab_prompt: Optional[str] = Form(None, description="Optional labels for open-vocabulary detection")
):
    """Recommend the best runnable model for an uploaded image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    is_valid, error_msg, image = validate_image(content, settings.MAX_UPLOAD_SIZE)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    standard_model = detection_service.auto_select_model(image)
    recommended_model = "yolov8s-world" if "yolov8s-world" in AVAILABLE_MODELS else standard_model
    recommendation = {
        "recommended_model": recommended_model,
        "mode": "openvocabulary" if recommended_model == "yolov8s-world" else "standard",
        "reason": (
            "Open-vocabulary detector selected for broad/unknown object coverage"
            if recommended_model == "yolov8s-world"
            else "Best standard detector for this image size and complexity"
        ),
        "open_vocabulary_available": "yolov8s-world" in AVAILABLE_MODELS,
        "open_vocabulary_classes": parse_open_vocab_prompt(open_vocab_prompt),
    }

    if "yolov8s-world" in AVAILABLE_MODELS:
        recommendation["fallback_model"] = standard_model
        recommendation["fallback_reason"] = (
            "Used automatically if open-vocabulary detection finds no objects"
        )

    return recommendation


@router.post("/detect/compare", response_model=DetectionCompareResponse)
async def compare_models(
    request: DetectionCompareRequest,
    file: UploadFile = File(..., description="Image file for comparison")
):
    """
    Compare multiple models on the same image
    
    Returns detection results from each model with performance metrics
    """
    # Validate file
    content = await file.read()
    is_valid, error_msg, image = validate_image(content, settings.MAX_UPLOAD_SIZE)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Validate models
    for model_id in request.models:
        if model_id not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_id}. Available: {list(AVAILABLE_MODELS.keys())}"
            )
        if not AVAILABLE_MODELS[model_id].get("runnable", True):
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' is catalog-only and is not runnable by the current inference backend"
            )
    
    try:
        # Run comparison
        comparison = detection_service.compare_models(
            image_source=image,
            model_ids=request.models,
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold
        )
        
        return comparison
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.post("/detect/compare-all")
async def compare_all_models(
    file: UploadFile = File(..., description="Image file for model comparison"),
    confidence_threshold: float = Form(default=0.25, ge=0, le=1),
    iou_threshold: float = Form(default=0.45, ge=0, le=1),
    open_vocab_prompt: Optional[str] = Form(None),
):
    """Run the uploaded image through every required runnable model."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    is_valid, error_msg, image = validate_image(content, settings.MAX_UPLOAD_SIZE)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    open_vocab_classes = parse_open_vocab_prompt(open_vocab_prompt)
    model_ids = [
        model_id
        for model_id, config in AVAILABLE_MODELS.items()
        if config.get("runnable", False)
    ]

    results = []
    for model_id in model_ids:
        config = AVAILABLE_MODELS[model_id]
        try:
            result = detection_service.detect_image(
                image_source=image,
                model_id=model_id,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                draw_boxes=False,
                return_base64=False,
                open_vocab_classes=open_vocab_classes,
            )
            classes = sorted({det["class_name"] for det in result["detections"]})
            avg_confidence = (
                round(sum(det["confidence"] for det in result["detections"]) / len(result["detections"]), 4)
                if result["detections"]
                else 0
            )
            results.append({
                "model_id": model_id,
                "name": config["name"],
                "family": config.get("family", config["type"]),
                "type": config["type"],
                "num_detections": result["num_detections"],
                "inference_time_ms": result["inference_time_ms"],
                "fps": result["fps"],
                "device_used": result["device_used"],
                "avg_confidence": avg_confidence,
                "classes": classes,
                "error": None,
            })
        except Exception as e:
            logger.error("Comparison failed for %s: %s", model_id, e)
            results.append({
                "model_id": model_id,
                "name": config["name"],
                "family": config.get("family", config["type"]),
                "type": config["type"],
                "num_detections": 0,
                "inference_time_ms": 0,
                "fps": 0,
                "device_used": "error",
                "avg_confidence": 0,
                "classes": [],
                "error": str(e),
            })

    successful = [item for item in results if not item["error"]]
    best_detection = max(successful, key=lambda item: item["num_detections"], default=None)
    fastest = min(
        [item for item in successful if item["inference_time_ms"] > 0],
        key=lambda item: item["inference_time_ms"],
        default=None,
    )

    return {
        "models": results,
        "summary": {
            "total_models": len(results),
            "successful_models": len(successful),
            "best_detection_model": best_detection["model_id"] if best_detection else None,
            "fastest_model": fastest["model_id"] if fastest else None,
        },
    }


@router.get("/recommend")
async def get_model_recommendation(
    use_case: str = Query(default="realtime", description="Use case: realtime, accuracy, balanced")
):
    """Get model recommendation based on use case"""
    manager = ModelManager()
    recommended = manager.recommend_model(use_case)
    model_info = manager.get_model_info(recommended)
    
    return {
        "recommended_model": recommended,
        "model_info": model_info,
        "use_case": use_case,
        "system_capabilities": manager.get_system_capabilities()
    }


@router.get("/system-info")
async def get_system_information():
    """Get detailed system information for diagnostics"""
    info = get_system_info()
    manager = ModelManager()
    
    return {
        **info,
        "loaded_models": manager.get_loaded_models(),
        "memory_usage": manager.get_memory_usage(),
        "capabilities": manager.get_system_capabilities()
    }


