"""
Model management API routes
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ..services.model_manager import ModelManager
from ..config import AVAILABLE_MODELS
from ..schemas.common import ModelInfo

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("/")
async def list_all_models():
    """Get all available models with their details"""
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
            "is_loaded": model_info.get("is_loaded", False) if model_info else False,
            "loaded_at": model_info.get("loaded_at") if model_info else None,
            "device": model_info.get("device") if model_info else None
        })
    
    return {
        "models": models,
        "total": len(models),
        "by_type": {
            "realtime": [m for m in models if m["type"] == "realtime"],
            "research": [m for m in models if m["type"] == "research"],
            "openvocabulary": [m for m in models if m["type"] == "openvocabulary"]
        },
        "runnable": [m for m in models if m["runnable"]],
        "catalog_only": [m for m in models if not m["runnable"]]
    }


@router.get("/recommend")
async def recommend_model(use_case: str = "realtime"):
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


@router.get("/system-capabilities")
async def get_system_capabilities():
    """Get system capabilities and model recommendations"""
    manager = ModelManager()
    return manager.get_system_capabilities()


@router.get("/loaded")
async def get_loaded_models():
    """Get list of currently loaded models"""
    manager = ModelManager()
    loaded = manager.get_loaded_models()
    return {
        "loaded_models": loaded,
        "count": len(loaded)
    }


@router.post("/unload/{model_id}")
async def unload_model(model_id: str):
    """Unload a specific model from memory"""
    manager = ModelManager()
    
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    success = manager.unload_model(model_id)
    if success:
        return {"message": f"Model '{model_id}' unloaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' was not loaded")


@router.post("/unload-all")
async def unload_all_models():
    """Unload all models from memory"""
    manager = ModelManager()
    manager.unload_all_models()
    return {"message": "All models unloaded successfully"}


@router.get("/memory-usage")
async def get_memory_usage():
    """Get current memory usage statistics"""
    manager = ModelManager()
    return manager.get_memory_usage()
