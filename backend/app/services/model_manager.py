"""
Model Manager - Handles model loading, caching, and device management
Supports YOLOv8 models with automatic CPU/GPU detection
"""

import torch
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path
import psutil
import platform
from datetime import datetime

from ..config import settings, AVAILABLE_MODELS, COCO_CLASSES
from ..utils.system import get_system_info

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages AI model loading, caching, and inference device selection
    Handles automatic CPU/GPU detection and fallback mechanisms
    """
    
    _instance = None
    _models: Dict[str, Any] = {}
    _model_metadata: Dict[str, Dict] = {}
    
    def __new__(cls):
        """Singleton pattern for model manager"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = self._detect_device()
        self.cuda_available = torch.cuda.is_available()
        self.gpu_name = None
        self.gpu_memory = None
        
        if self.cuda_available:
            try:
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                logger.info(f"GPU detected: {self.gpu_name} ({self.gpu_memory:.1f} GB)")
            except Exception as e:
                logger.warning(f"Could not get GPU details: {e}")
        
        self.system_info = get_system_info()
        self._initialized = True
        logger.info(f"Model Manager initialized - Device: {self.device}")
    
    def _detect_device(self) -> str:
        """
        Automatically detect and select the best compute device
        Returns: 'cuda' if GPU available, otherwise 'cpu'
        """
        if settings.DEVICE == "cpu":
            logger.info("CPU mode forced by configuration")
            return "cpu"
        
        if torch.cuda.is_available():
            logger.info("CUDA available - using GPU acceleration")
            return "cuda"
        
        logger.info("CUDA not available - using CPU")
        return "cpu"
    
    def get_device(self) -> torch.device:
        """Get PyTorch device object"""
        return torch.device(self.device)
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities for model recommendation"""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        capabilities = {
            "device": self.device,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory,
            "cpu_count": psutil.cpu_count(),
            "ram_gb": ram_gb,
            "platform": platform.system(),
            "supports_fp16": self.cuda_available and torch.cuda.get_device_capability()[0] >= 7 if self.cuda_available else False,
            "recommended_models": []
        }
        
        # Recommend models based on capabilities
        if ram_gb >= 16 and self.cuda_available and self.gpu_memory >= 8:
            capabilities["recommended_models"] = ["yolov8x", "yolov8l", "yolov8m", "yolov8s", "yolov8n"]
        elif ram_gb >= 8 and self.cuda_available:
            capabilities["recommended_models"] = ["yolov8m", "yolov8s", "yolov8n"]
        elif ram_gb >= 4:
            capabilities["recommended_models"] = ["yolov8s", "yolov8n"]
        else:
            capabilities["recommended_models"] = ["yolov8n"]
        
        return capabilities
    
    def load_model(self, model_id: str, force_reload: bool = False) -> Any:
        """
        Load a detection model with caching
        
        Args:
            model_id: Model identifier (e.g., 'yolov8n')
            force_reload: Force reload even if cached
        
        Returns:
            Loaded model
        """
        # Check cache first
        if model_id in self._models and not force_reload:
            logger.info(f"Using cached model: {model_id}")
            return self._models[model_id]
        
        if model_id not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(AVAILABLE_MODELS.keys())}")
        
        model_config = AVAILABLE_MODELS[model_id]
        if not model_config.get("runnable", True):
            raise ValueError(
                f"Model '{model_id}' is catalog-only. It is listed for comparison, "
                "but the current inference runtime does not load that framework yet."
            )

        weights_path = self._resolve_weights_path(model_config["weights"])
        
        logger.info(f"Loading model: {model_id} ({weights_path})")
        
        try:
            # Import YOLO here to avoid circular imports
            from ultralytics import YOLO, YOLOWorld
            
            # Load model
            if model_config.get("backend") == "yolo-world":
                model = YOLOWorld(str(weights_path))
            else:
                model = YOLO(str(weights_path))
            
            # Move to appropriate device
            model.to(self.device)
            
            # Set model to evaluation mode
            model.eval()
            
            # Cache the model
            self._models[model_id] = model
            self._model_metadata[model_id] = {
                "loaded_at": datetime.utcnow().isoformat(),
                "device": self.device,
                "config": model_config
            }
            
            logger.info(f"Model loaded successfully: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")

    def _resolve_weights_path(self, weights: str) -> Path:
        """Resolve model weights from configured, project, or working paths."""
        weights_path = Path(weights)
        if weights_path.is_absolute() or weights_path.exists():
            return weights_path

        candidates = [
            settings.weights_path / weights,
            settings.project_root / weights,
            settings.project_root / "backend" / "weights" / weights,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Let Ultralytics handle remote/default resolution as a final fallback.
        return weights_path
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        if model_id in self._models:
            del self._models[model_id]
            if model_id in self._model_metadata:
                del self._model_metadata[model_id]
            logger.info(f"Model unloaded: {model_id}")
            return True
        return False
    
    def unload_all_models(self):
        """Unload all loaded models"""
        self._models.clear()
        self._model_metadata.clear()
        logger.info("All models unloaded")
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self._models.keys())
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_id not in AVAILABLE_MODELS:
            return None
        
        config = AVAILABLE_MODELS[model_id]
        metadata = self._model_metadata.get(model_id, {})
        
        return {
            "id": model_id,
            "name": config["name"],
            "type": config["type"],
            "family": config.get("family", config["type"]),
            "backend": config.get("backend", "ultralytics"),
            "runnable": config.get("runnable", True),
            "description": config["description"],
            "input_size": config["input_size"],
            "recommended_for": config["recommended_for"],
            "is_loaded": model_id in self._models,
            "loaded_at": metadata.get("loaded_at"),
            "device": metadata.get("device")
        }
    
    def get_all_model_info(self) -> List[Dict[str, Any]]:
        """Get information about all available models"""
        return [self.get_model_info(model_id) for model_id in AVAILABLE_MODELS]
    
    def recommend_model(self, use_case: str = "realtime") -> str:
        """
        Recommend the best model for a given use case
        
        Args:
            use_case: 'realtime', 'accuracy', 'balanced', 'cpu', 'mobile'
        
        Returns:
            Recommended model ID
        """
        capabilities = self.get_system_capabilities()
        
        # Filter models by use case
        suitable_models = []
        for model_id, config in AVAILABLE_MODELS.items():
            if use_case in config["recommended_for"]:
                suitable_models.append(model_id)
        
        if not suitable_models:
            # Fallback to nano model
            return "yolov8n"
        
        # If multiple models match, prefer based on system capabilities
        if len(suitable_models) > 1:
            if use_case == "accuracy":
                # Prefer larger models for accuracy
                priority = ["yolov8x", "yolov8l", "yolov8m", "yolov8s", "yolov8n"]
            elif use_case == "realtime" or use_case == "mobile":
                # Prefer smaller models for speed
                priority = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
            else:
                # Balanced
                priority = ["yolov8s", "yolov8m", "yolov8n", "yolov8l", "yolov8x"]
            
            for model in priority:
                if model in suitable_models:
                    return model
        
        return suitable_models[0]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        if self.cuda_available:
            try:
                memory_info["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / (1024**3)
                memory_info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
                memory_info["gpu_memory_total_gb"] = self.gpu_memory
            except:
                pass
        
        return memory_info
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleared")
    
    def get_class_names(self) -> List[str]:
        """Get COCO class names"""
        return COCO_CLASSES
