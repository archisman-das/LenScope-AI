"""
Configuration settings for LenScope AI Backend
Loads environment variables and provides type-safe configuration
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application Settings
    APP_NAME: str = "LenScope AI Dashboard"
    DEBUG: bool = Field(default=False, validate_default=False)
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5501",
        "http://127.0.0.1:5501",
        "http://localhost:5502",
        "http://127.0.0.1:5502",
        "http://localhost:5503",
        "http://127.0.0.1:5503",
        "*"  # Allow all for development
    ]
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./lenscope.db"
    
    # AI/ML Settings
    DEFAULT_MODEL: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    MAX_DETECTIONS: int = 100
    DEVICE: str = "auto"  # auto, cpu, cuda
    
    # Image Settings
    MAX_UPLOAD_SIZE: int = -1  # Unlimited (-1 means no limit)
    ALLOWED_EXTENSIONS: str = "jpg,jpeg,png,bmp,webp,gif,tiff,tif,ico,svg,avif,jfif"
    UPLOAD_DIR: str = "./uploads"
    WEIGHTS_DIR: str = "./weights"
    
    # ONNX Settings
    USE_ONNX: bool = False
    ONNX_DEVICE: str = "CPU"  # CPU, CUDA, TensorRT
    
    # Optimization
    USE_FP16: bool = False
    USE_INT8: bool = False
    AUTO_DOWNSCALE: bool = True
    MAX_RESOLUTION: int = 1280
    ADAPTIVE_ROUTING_ENABLED: bool = True
    
    # Webcam Settings
    WEBCAM_FPS: int = 30
    WEBCAM_WIDTH: int = 640
    WEBCAM_HEIGHT: int = 480
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/lenscope.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        # Override DEBUG from environment if it's not a valid boolean
        debug_val = os.environ.get("DEBUG")
        if debug_val is not None and debug_val.lower() not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            # Remove the invalid DEBUG value so the default is used
            del os.environ["DEBUG"]
        super().__init__(**kwargs)
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Get allowed file extensions as a list"""
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")]
    
    @property
    def project_root(self) -> Path:
        """Get project root directory"""
        return Path(__file__).parent.parent.parent
    
    @property
    def upload_path(self) -> Path:
        """Get upload directory path"""
        return self.project_root / self.UPLOAD_DIR.lstrip("./")
    
    @property
    def weights_path(self) -> Path:
        """Get weights directory path"""
        return self.project_root / self.WEIGHTS_DIR.lstrip("./")
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.weights_path.mkdir(parents=True, exist_ok=True)
        log_dir = self.project_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)


# Available models configuration
AVAILABLE_MODELS = {
    # Real-time YOLOv8 models
    "yolov8n": {
        "name": "YOLOv8n",
        "type": "realtime",
        "weights": "yolov8n.pt",
        "input_size": 640,
        "description": "Nano - Fastest, smallest model",
        "recommended_for": ["realtime", "mobile", "cpu", "low-resources"],
        "min_ram_gb": 2,
        "min_gpu_memory_gb": 0,
        "expected_fps_cpu": 30,
        "expected_fps_gpu": 60,
        "accuracy_score": 0.65,
        "best_for": "Plain text, simple scenes, low-resource devices"
    },
    "yolov8s": {
        "name": "YOLOv8s",
        "type": "realtime",
        "weights": "yolov8s.pt",
        "input_size": 640,
        "description": "Small - Fast with good accuracy",
        "recommended_for": ["realtime", "balanced"],
        "min_ram_gb": 4,
        "min_gpu_memory_gb": 2,
        "expected_fps_cpu": 20,
        "expected_fps_gpu": 45,
        "accuracy_score": 0.72,
        "best_for": "General purpose detection with moderate complexity"
    },
    "yolov8m": {
        "name": "YOLOv8m",
        "type": "realtime",
        "weights": "yolov8m.pt",
        "input_size": 640,
        "description": "Medium - Balanced speed and accuracy",
        "recommended_for": ["accuracy", "gpu", "balanced"],
        "min_ram_gb": 6,
        "min_gpu_memory_gb": 4,
        "expected_fps_cpu": 10,
        "expected_fps_gpu": 30,
        "accuracy_score": 0.78,
        "best_for": "Medium complexity scenes with multiple objects"
    },
    "yolov8l": {
        "name": "YOLOv8l",
        "type": "realtime",
        "weights": "yolov8l.pt",
        "input_size": 640,
        "description": "Large - High accuracy, slower",
        "recommended_for": ["accuracy", "gpu"],
        "min_ram_gb": 8,
        "min_gpu_memory_gb": 6,
        "expected_fps_cpu": 5,
        "expected_fps_gpu": 20,
        "accuracy_score": 0.82,
        "best_for": "High accuracy requirements with GPU available"
    },
    "yolov8x": {
        "name": "YOLOv8x",
        "type": "realtime",
        "weights": "yolov8x.pt",
        "input_size": 640,
        "description": "Extra Large - Highest accuracy",
        "recommended_for": ["accuracy", "gpu"],
        "min_ram_gb": 12,
        "min_gpu_memory_gb": 8,
        "expected_fps_cpu": 3,
        "expected_fps_gpu": 15,
        "accuracy_score": 0.85,
        "best_for": "Maximum accuracy, complex scenes with many objects"
    },
    # RT-DETR (Real-Time Detection Transformer) - Best for crowded scenes
    "rt-detr-l": {
        "name": "RT-DETR-L",
        "type": "realtime",
        "weights": "rtdetr-l.pt",
        "input_size": 640,
        "description": "Real-Time Detection Transformer - Large (Transformer-based)",
        "recommended_for": ["accuracy", "gpu", "realtime", "crowded-scenes"],
        "min_ram_gb": 8,
        "min_gpu_memory_gb": 6,
        "expected_fps_cpu": 8,
        "expected_fps_gpu": 25,
        "accuracy_score": 0.80,
        "best_for": "Crowded scenes with many overlapping objects"
    },
    "rt-detr-x": {
        "name": "RT-DETR-X",
        "type": "realtime",
        "weights": "rtdetr-x.pt",
        "input_size": 640,
        "description": "Real-Time Detection Transformer - XLarge (Highest accuracy)",
        "recommended_for": ["accuracy", "gpu", "crowded-scenes"],
        "min_ram_gb": 12,
        "min_gpu_memory_gb": 8,
        "expected_fps_cpu": 5,
        "expected_fps_gpu": 18,
        "accuracy_score": 0.84,
        "best_for": "Extremely crowded scenes requiring transformer attention"
    },
    # MobileNet SSD - Ultra-lightweight for edge devices
    "mobilenet-ssd": {
        "name": "MobileNet SSD",
        "type": "realtime",
        "weights": "mobilenet-ssd.caffemodel",
        "input_size": 300,
        "description": "Ultra-lightweight model for mobile and edge devices",
        "recommended_for": ["mobile", "cpu", "low-resources", "realtime", "edge"],
        "min_ram_gb": 1,
        "min_gpu_memory_gb": 0,
        "expected_fps_cpu": 50,
        "expected_fps_gpu": 80,
        "accuracy_score": 0.55,
        "best_for": "Low-resource devices, mobile phones, embedded systems"
    },
    # Research/Comparison models - Faster R-CNN variants
    "faster-rcnn": {
        "name": "Faster R-CNN",
        "type": "research",
        "weights": "yolov8m.pt",  # Using YOLO as proxy when weights unavailable
        "input_size": 800,
        "description": "Two-stage detector - High accuracy",
        "recommended_for": ["research", "comparison"],
        "min_ram_gb": 8,
        "min_gpu_memory_gb": 4,
        "expected_fps_cpu": 2,
        "expected_fps_gpu": 10,
        "accuracy_score": 0.80,
        "best_for": "Research and comparison with two-stage detectors"
    },
    "faster-rcnn-r50": {
        "name": "Faster R-CNN R50",
        "type": "research",
        "weights": "yolov8s.pt",  # Using YOLO as proxy
        "input_size": 800,
        "description": "Faster R-CNN with ResNet-50 backbone",
        "recommended_for": ["research", "comparison", "balanced"],
        "min_ram_gb": 6,
        "min_gpu_memory_gb": 4,
        "expected_fps_cpu": 3,
        "expected_fps_gpu": 12,
        "accuracy_score": 0.78,
        "best_for": "Research with balanced speed and accuracy"
    },
    "faster-rcnn-r101": {
        "name": "Faster R-CNN R101",
        "type": "research",
        "weights": "yolov8m.pt",  # Using YOLO as proxy
        "input_size": 800,
        "description": "Faster R-CNN with ResNet-101 backbone - Higher accuracy",
        "recommended_for": ["research", "comparison", "accuracy"],
        "min_ram_gb": 10,
        "min_gpu_memory_gb": 6,
        "expected_fps_cpu": 2,
        "expected_fps_gpu": 8,
        "accuracy_score": 0.82,
        "best_for": "Research requiring highest accuracy two-stage detection"
    },
    # Research/Comparison models - RetinaNet
    "retinanet": {
        "name": "RetinaNet",
        "type": "research",
        "weights": "yolov8s.pt",  # Using YOLO as proxy
        "input_size": 800,
        "description": "Focal loss based detector",
        "recommended_for": ["research", "comparison"],
        "min_ram_gb": 6,
        "min_gpu_memory_gb": 4,
        "expected_fps_cpu": 3,
        "expected_fps_gpu": 12,
        "accuracy_score": 0.77,
        "best_for": "Research on focal loss and hard example mining"
    },
    # Open vocabulary models
    "grounding-dino": {
        "name": "Grounding DINO",
        "type": "openvocabulary",
        "weights": "groundingdino_swint_ogc.pth",
        "input_size": 800,
        "description": "Open vocabulary detection with text prompts",
        "recommended_for": ["openvocabulary", "gpu"],
        "min_ram_gb": 16,
        "min_gpu_memory_gb": 8,
        "expected_fps_cpu": 1,
        "expected_fps_gpu": 5,
        "accuracy_score": 0.75,
        "best_for": "Custom object detection with text prompts"
    }
}


def _model_entry(
    name: str,
    family: str,
    model_type: str,
    weights: str,
    input_size: int,
    description: str,
    recommended_for: List[str],
    min_ram_gb: int,
    min_gpu_memory_gb: int,
    expected_fps_cpu: int,
    expected_fps_gpu: int,
    accuracy_score: float,
    best_for: str,
    backend: str = "catalog",
    runnable: bool = False,
) -> dict:
    return {
        "name": name,
        "family": family,
        "type": model_type,
        "weights": weights,
        "input_size": input_size,
        "description": description,
        "recommended_for": recommended_for,
        "min_ram_gb": min_ram_gb,
        "min_gpu_memory_gb": min_gpu_memory_gb,
        "expected_fps_cpu": expected_fps_cpu,
        "expected_fps_gpu": expected_fps_gpu,
        "accuracy_score": accuracy_score,
        "best_for": best_for,
        "backend": backend,
        "runnable": runnable,
    }


def _variant_family(
    prefix: str,
    display_prefix: str,
    family: str,
    variants: List[tuple],
    description: str,
    best_for: str,
    backend: str = "catalog",
    runnable: bool = False,
    weights_template: str = "{id}.pt",
    model_type: str = "research",
) -> dict:
    models = {}
    for suffix, label, input_size, ram, gpu, fps_cpu, fps_gpu, accuracy, tags in variants:
        model_id = f"{prefix}{suffix}"
        models[model_id] = _model_entry(
            name=f"{display_prefix}{label}",
            family=family,
            model_type=model_type,
            weights=weights_template.format(id=model_id),
            input_size=input_size,
            description=description,
            recommended_for=tags,
            min_ram_gb=ram,
            min_gpu_memory_gb=gpu,
            expected_fps_cpu=fps_cpu,
            expected_fps_gpu=fps_gpu,
            accuracy_score=accuracy,
            best_for=best_for,
            backend=backend,
            runnable=runnable,
        )
    return models


# Mark the original YOLOv8 and RT-DETR entries as runnable by the current
# Ultralytics inference path.
for _model_id in list(AVAILABLE_MODELS):
    AVAILABLE_MODELS[_model_id].setdefault("family", "YOLOv8" if _model_id.startswith("yolov8") else "Object Detection")
    AVAILABLE_MODELS[_model_id].setdefault("backend", "ultralytics")
    AVAILABLE_MODELS[_model_id].setdefault("runnable", _model_id.startswith("yolov8") or _model_id.startswith("rt-detr"))


AVAILABLE_MODELS.update({
    # Ultralytics YOLO object detection families
    **_variant_family(
        "yolo11",
        "YOLO11",
        "YOLO",
        [
            ("n", "n", 640, 2, 0, 32, 68, 0.69, ["realtime", "mobile", "cpu", "low-resources"]),
            ("s", "s", 640, 4, 2, 24, 54, 0.75, ["realtime", "balanced"]),
            ("m", "m", 640, 6, 4, 13, 36, 0.81, ["balanced", "accuracy", "gpu"]),
            ("l", "l", 640, 8, 6, 7, 24, 0.84, ["accuracy", "gpu"]),
            ("x", "x", 640, 12, 8, 4, 17, 0.87, ["accuracy", "gpu", "research"]),
        ],
        "Current Ultralytics YOLO detector family with nano through extra-large variants.",
        "General purpose detection with a modern speed/accuracy balance",
        backend="ultralytics",
        runnable=True,
        model_type="realtime",
    ),
    **_variant_family(
        "yolov10",
        "YOLOv10",
        "YOLO",
        [
            ("n", "n", 640, 2, 0, 31, 66, 0.68, ["realtime", "mobile", "cpu"]),
            ("s", "s", 640, 4, 2, 23, 52, 0.74, ["realtime", "balanced"]),
            ("m", "m", 640, 6, 4, 12, 34, 0.80, ["balanced", "accuracy", "gpu"]),
            ("b", "b", 640, 8, 5, 9, 29, 0.82, ["balanced", "accuracy", "gpu"]),
            ("l", "l", 640, 10, 6, 6, 22, 0.84, ["accuracy", "gpu"]),
            ("x", "x", 640, 12, 8, 4, 16, 0.86, ["accuracy", "gpu", "research"]),
        ],
        "YOLOv10 detector variants focused on efficient end-to-end detection.",
        "Low-latency detection and model comparison",
        backend="ultralytics",
        runnable=True,
        model_type="realtime",
    ),
    **_variant_family(
        "yolov9",
        "YOLOv9",
        "YOLO",
        [
            ("t", "t", 640, 2, 0, 30, 64, 0.67, ["realtime", "mobile", "cpu"]),
            ("s", "s", 640, 4, 2, 22, 50, 0.73, ["realtime", "balanced"]),
            ("m", "m", 640, 6, 4, 11, 32, 0.79, ["balanced", "accuracy", "gpu"]),
            ("c", "c", 640, 8, 6, 7, 24, 0.83, ["accuracy", "gpu"]),
            ("e", "e", 640, 12, 8, 4, 16, 0.86, ["accuracy", "gpu", "research"]),
        ],
        "YOLOv9 detector variants for real-time and high-accuracy workloads.",
        "YOLO-family benchmarking across speed and accuracy profiles",
        backend="ultralytics",
        runnable=True,
        model_type="realtime",
    ),
    **_variant_family(
        "yolov5",
        "YOLOv5",
        "YOLO",
        [
            ("n", "n", 640, 2, 0, 36, 72, 0.62, ["realtime", "mobile", "cpu", "low-resources"]),
            ("s", "s", 640, 4, 2, 26, 56, 0.69, ["realtime", "balanced"]),
            ("m", "m", 640, 6, 4, 14, 38, 0.75, ["balanced", "gpu"]),
            ("l", "l", 640, 8, 6, 8, 25, 0.79, ["accuracy", "gpu"]),
            ("x", "x", 640, 12, 8, 5, 18, 0.82, ["accuracy", "gpu", "research"]),
            ("n6", "n6", 1280, 4, 2, 16, 42, 0.68, ["high-resolution", "realtime"]),
            ("s6", "s6", 1280, 6, 4, 10, 31, 0.74, ["high-resolution", "balanced"]),
            ("m6", "m6", 1280, 8, 6, 6, 22, 0.79, ["high-resolution", "accuracy", "gpu"]),
            ("l6", "l6", 1280, 12, 8, 4, 16, 0.83, ["high-resolution", "accuracy", "gpu"]),
            ("x6", "x6", 1280, 16, 10, 3, 12, 0.85, ["high-resolution", "accuracy", "research"]),
        ],
        "YOLOv5 detector family, including standard and high-resolution P6 variants.",
        "Legacy YOLO baselines, edge devices, and high-resolution comparison",
        backend="ultralytics",
        runnable=True,
        model_type="realtime",
    ),
    **_variant_family(
        "yolov3",
        "YOLOv3",
        "YOLO",
        [
            ("tiny", "-tiny", 416, 2, 0, 42, 80, 0.45, ["realtime", "mobile", "cpu", "legacy"]),
            ("", "", 416, 4, 2, 14, 38, 0.60, ["legacy", "balanced"]),
            ("spp", "-spp", 608, 6, 4, 8, 24, 0.64, ["legacy", "accuracy", "gpu"]),
        ],
        "Classic YOLOv3 detector variants for historical baselines.",
        "Legacy benchmarks and lightweight comparison",
        backend="ultralytics",
        runnable=True,
        model_type="research",
    ),
    **_variant_family(
        "yolov4",
        "YOLOv4",
        "YOLO",
        [
            ("tiny", "-tiny", 416, 2, 0, 38, 74, 0.52, ["realtime", "mobile", "legacy"]),
            ("", "", 608, 6, 4, 9, 28, 0.70, ["legacy", "balanced", "gpu"]),
            ("csp", "-CSP", 608, 8, 6, 6, 22, 0.74, ["legacy", "accuracy", "gpu"]),
            ("x", "-x", 640, 10, 8, 4, 16, 0.77, ["legacy", "accuracy", "research"]),
        ],
        "YOLOv4 detector variants from the Darknet-era YOLO family.",
        "Legacy Darknet YOLO benchmarks and comparison",
        model_type="research",
    ),
    **_variant_family(
        "yolov6",
        "YOLOv6",
        "YOLO",
        [
            ("n", "n", 640, 2, 0, 34, 70, 0.64, ["realtime", "mobile", "cpu"]),
            ("s", "s", 640, 4, 2, 24, 54, 0.71, ["realtime", "balanced"]),
            ("m", "m", 640, 6, 4, 13, 36, 0.77, ["balanced", "gpu"]),
            ("l", "l", 640, 8, 6, 7, 24, 0.81, ["accuracy", "gpu"]),
            ("l6", "l6", 1280, 12, 8, 4, 15, 0.84, ["accuracy", "gpu", "high-resolution"]),
        ],
        "YOLOv6 detector family with nano through high-resolution variants.",
        "Industrial YOLO comparison and high-resolution detection",
        model_type="research",
    ),
    **_variant_family(
        "yolov7",
        "YOLOv7",
        "YOLO",
        [
            ("tiny", "-tiny", 416, 2, 0, 36, 72, 0.58, ["realtime", "mobile", "legacy"]),
            ("", "", 640, 6, 4, 14, 38, 0.76, ["balanced", "gpu", "legacy"]),
            ("x", "-x", 640, 8, 6, 8, 26, 0.80, ["accuracy", "gpu", "legacy"]),
            ("w6", "-w6", 1280, 10, 8, 5, 18, 0.82, ["accuracy", "gpu", "high-resolution"]),
            ("e6", "-e6", 1280, 12, 10, 4, 14, 0.84, ["accuracy", "gpu", "high-resolution"]),
            ("d6", "-d6", 1280, 14, 12, 3, 11, 0.85, ["accuracy", "gpu", "research"]),
            ("e6e", "-e6e", 1280, 16, 14, 2, 9, 0.86, ["accuracy", "gpu", "research"]),
        ],
        "YOLOv7 detector family including tiny, x, and high-resolution E/D variants.",
        "Legacy high-performance YOLO benchmarking",
        model_type="research",
    ),

    # Transformer and real-time transformer detectors
    "rtdetr-r18": _model_entry("RT-DETR-R18", "RT-DETR", "realtime", "rtdetr-r18.pt", 640, "Compact real-time detection transformer.", ["realtime", "gpu", "balanced"], 4, 2, 14, 36, 0.72, "Real-time transformer baseline", "ultralytics", True),
    "rtdetr-r34": _model_entry("RT-DETR-R34", "RT-DETR", "realtime", "rtdetr-r34.pt", 640, "Mid-size real-time detection transformer.", ["realtime", "gpu", "balanced"], 6, 4, 10, 30, 0.76, "Balanced transformer detection", "ultralytics", True),
    "rtdetr-r50": _model_entry("RT-DETR-R50", "RT-DETR", "realtime", "rtdetr-r50.pt", 640, "ResNet-50 real-time detection transformer.", ["accuracy", "gpu", "crowded-scenes"], 8, 6, 7, 24, 0.80, "Crowded scenes and overlapping objects", "ultralytics", True),
    "detr-r50": _model_entry("DETR-R50", "DETR", "research", "detr-r50.pth", 800, "Original Detection Transformer with ResNet-50 backbone.", ["research", "comparison", "accuracy"], 8, 6, 1, 8, 0.76, "Transformer research baseline"),
    "detr-r101": _model_entry("DETR-R101", "DETR", "research", "detr-r101.pth", 800, "Original Detection Transformer with ResNet-101 backbone.", ["research", "comparison", "accuracy"], 10, 8, 1, 6, 0.79, "Higher-capacity transformer baseline"),
    "deformable-detr-r50": _model_entry("Deformable DETR-R50", "Deformable DETR", "research", "deformable-detr-r50.pth", 800, "DETR variant with deformable attention for faster convergence and better small-object handling.", ["research", "accuracy", "crowded-scenes"], 10, 8, 1, 7, 0.82, "Complex scenes with multi-scale objects"),
    "dino-r50": _model_entry("DINO-R50", "DINO", "research", "dino-r50.pth", 800, "Transformer detector using improved denoising training.", ["research", "accuracy"], 12, 8, 1, 6, 0.84, "High-accuracy transformer comparison"),

    # Two-stage and dense detector families
    "faster-rcnn-r50-fpn": _model_entry("Faster R-CNN R50-FPN", "Faster R-CNN", "research", "faster-rcnn-r50-fpn.pth", 800, "Two-stage detector with ResNet-50 FPN backbone.", ["research", "comparison", "balanced"], 6, 4, 3, 12, 0.78, "Reliable two-stage detection baseline"),
    "faster-rcnn-r101-fpn": _model_entry("Faster R-CNN R101-FPN", "Faster R-CNN", "research", "faster-rcnn-r101-fpn.pth", 800, "Two-stage detector with ResNet-101 FPN backbone.", ["research", "comparison", "accuracy"], 10, 6, 2, 8, 0.82, "Higher-accuracy two-stage detection"),
    "faster-rcnn-x101-fpn": _model_entry("Faster R-CNN X101-FPN", "Faster R-CNN", "research", "faster-rcnn-x101-fpn.pth", 800, "Two-stage detector with ResNeXt-101 FPN backbone.", ["research", "accuracy", "gpu"], 12, 8, 1, 6, 0.84, "Maximum two-stage accuracy"),
    "mask-rcnn-r50-fpn": _model_entry("Mask R-CNN R50-FPN", "Mask R-CNN", "research", "mask-rcnn-r50-fpn.pth", 800, "Instance segmentation detector with bounding-box output support.", ["research", "segmentation", "accuracy"], 8, 6, 2, 9, 0.79, "Detection plus instance segmentation workflows"),
    "mask-rcnn-r101-fpn": _model_entry("Mask R-CNN R101-FPN", "Mask R-CNN", "research", "mask-rcnn-r101-fpn.pth", 800, "Higher-capacity Mask R-CNN variant.", ["research", "segmentation", "accuracy"], 12, 8, 1, 6, 0.82, "High-accuracy segmentation-aware detection"),
    "cascade-rcnn-r50-fpn": _model_entry("Cascade R-CNN R50-FPN", "Cascade R-CNN", "research", "cascade-rcnn-r50-fpn.pth", 800, "Cascade detector that refines boxes across multiple stages.", ["research", "accuracy"], 10, 6, 2, 8, 0.83, "Precise localization"),
    "retinanet-r50-fpn": _model_entry("RetinaNet R50-FPN", "RetinaNet", "research", "retinanet-r50-fpn.pth", 800, "One-stage dense detector using focal loss.", ["research", "comparison", "balanced"], 6, 4, 4, 14, 0.77, "Dense one-stage detection baseline"),
    "retinanet-r101-fpn": _model_entry("RetinaNet R101-FPN", "RetinaNet", "research", "retinanet-r101-fpn.pth", 800, "Higher-capacity RetinaNet variant.", ["research", "comparison", "accuracy"], 10, 6, 3, 10, 0.80, "Dense detection with better accuracy"),
    "fcos-r50-fpn": _model_entry("FCOS R50-FPN", "FCOS", "research", "fcos-r50-fpn.pth", 800, "Anchor-free fully convolutional one-stage detector.", ["research", "comparison", "balanced"], 6, 4, 4, 14, 0.78, "Anchor-free detection baseline"),
    "centernet-r50": _model_entry("CenterNet R50", "CenterNet", "research", "centernet-r50.pth", 512, "Keypoint-style detector that predicts object centers.", ["research", "realtime", "comparison"], 4, 2, 12, 30, 0.72, "Fast anchor-free object center detection"),
    "ssd300-vgg16": _model_entry("SSD300 VGG16", "SSD", "research", "ssd300-vgg16.pth", 300, "Classic single-shot detector at 300px input size.", ["realtime", "legacy", "cpu"], 3, 0, 24, 50, 0.58, "Fast legacy single-shot detection"),
    "ssd512-vgg16": _model_entry("SSD512 VGG16", "SSD", "research", "ssd512-vgg16.pth", 512, "Classic single-shot detector at 512px input size.", ["balanced", "legacy"], 4, 2, 14, 34, 0.64, "Legacy detection with better small-object recall"),
    "ssdlite-mobilenetv3": _model_entry("SSDLite MobileNetV3", "SSDLite", "research", "ssdlite-mobilenetv3.pth", 320, "Mobile-optimized SSD variant.", ["mobile", "cpu", "edge", "realtime"], 2, 0, 38, 70, 0.57, "Mobile and edge deployments"),
    "efficientdet-d0": _model_entry("EfficientDet-D0", "EfficientDet", "research", "efficientdet-d0.pth", 512, "Small EfficientDet detector.", ["mobile", "cpu", "balanced"], 3, 1, 18, 42, 0.68, "Efficient detection on modest hardware"),
    "efficientdet-d1": _model_entry("EfficientDet-D1", "EfficientDet", "research", "efficientdet-d1.pth", 640, "EfficientDet variant with higher resolution.", ["balanced", "gpu"], 4, 2, 12, 34, 0.72, "Balanced efficient detection"),
    "efficientdet-d2": _model_entry("EfficientDet-D2", "EfficientDet", "research", "efficientdet-d2.pth", 768, "Mid-size EfficientDet detector.", ["balanced", "accuracy", "gpu"], 6, 4, 8, 25, 0.76, "Efficient high-resolution detection"),
    "efficientdet-d3": _model_entry("EfficientDet-D3", "EfficientDet", "research", "efficientdet-d3.pth", 896, "Larger EfficientDet detector.", ["accuracy", "gpu"], 8, 6, 5, 18, 0.79, "Accuracy-focused efficient detection"),
    "efficientdet-d4": _model_entry("EfficientDet-D4", "EfficientDet", "research", "efficientdet-d4.pth", 1024, "High-resolution EfficientDet detector.", ["accuracy", "gpu", "high-resolution"], 10, 8, 3, 12, 0.82, "High-resolution scenes"),
    "efficientdet-d5": _model_entry("EfficientDet-D5", "EfficientDet", "research", "efficientdet-d5.pth", 1280, "Large EfficientDet detector.", ["accuracy", "gpu", "high-resolution"], 12, 10, 2, 9, 0.84, "High-accuracy image analysis"),
    "efficientdet-d6": _model_entry("EfficientDet-D6", "EfficientDet", "research", "efficientdet-d6.pth", 1280, "Very large EfficientDet detector.", ["accuracy", "gpu", "research"], 16, 12, 1, 6, 0.85, "Offline high-accuracy comparison"),
    "efficientdet-d7": _model_entry("EfficientDet-D7", "EfficientDet", "research", "efficientdet-d7.pth", 1536, "Largest EfficientDet catalog entry.", ["accuracy", "gpu", "research"], 20, 16, 1, 4, 0.86, "Maximum EfficientDet accuracy"),

    # YOLO-adjacent research families
    "yolox-nano": _model_entry("YOLOX-Nano", "YOLOX", "research", "yolox-nano.pth", 416, "Ultra-light YOLOX variant.", ["mobile", "cpu", "realtime"], 2, 0, 40, 78, 0.55, "Very low-resource detection"),
    "yolox-tiny": _model_entry("YOLOX-Tiny", "YOLOX", "research", "yolox-tiny.pth", 416, "Tiny YOLOX variant.", ["mobile", "cpu", "realtime"], 2, 0, 34, 68, 0.61, "Fast lightweight detection"),
    "yolox-s": _model_entry("YOLOX-S", "YOLOX", "research", "yolox-s.pth", 640, "Small YOLOX detector.", ["realtime", "balanced"], 4, 2, 24, 54, 0.70, "YOLOX speed baseline"),
    "yolox-m": _model_entry("YOLOX-M", "YOLOX", "research", "yolox-m.pth", 640, "Medium YOLOX detector.", ["balanced", "gpu"], 6, 4, 13, 36, 0.76, "Balanced YOLOX comparison"),
    "yolox-l": _model_entry("YOLOX-L", "YOLOX", "research", "yolox-l.pth", 640, "Large YOLOX detector.", ["accuracy", "gpu"], 8, 6, 7, 24, 0.80, "Accuracy-focused YOLOX comparison"),
    "yolox-x": _model_entry("YOLOX-X", "YOLOX", "research", "yolox-x.pth", 640, "Extra-large YOLOX detector.", ["accuracy", "gpu", "research"], 12, 8, 4, 16, 0.83, "Maximum YOLOX accuracy"),
    "ppyoloe-s": _model_entry("PP-YOLOE-S", "PP-YOLOE", "research", "ppyoloe-s.pdparams", 640, "Small PP-YOLOE detector.", ["realtime", "balanced"], 4, 2, 22, 52, 0.71, "PaddleDetection YOLO comparison"),
    "ppyoloe-m": _model_entry("PP-YOLOE-M", "PP-YOLOE", "research", "ppyoloe-m.pdparams", 640, "Medium PP-YOLOE detector.", ["balanced", "gpu"], 6, 4, 12, 34, 0.77, "Balanced PaddleDetection comparison"),
    "ppyoloe-l": _model_entry("PP-YOLOE-L", "PP-YOLOE", "research", "ppyoloe-l.pdparams", 640, "Large PP-YOLOE detector.", ["accuracy", "gpu"], 8, 6, 7, 24, 0.81, "High-accuracy PaddleDetection comparison"),
    "ppyoloe-x": _model_entry("PP-YOLOE-X", "PP-YOLOE", "research", "ppyoloe-x.pdparams", 640, "Extra-large PP-YOLOE detector.", ["accuracy", "gpu", "research"], 12, 8, 4, 16, 0.84, "Maximum PP-YOLOE comparison"),

    # Open vocabulary detectors
    "grounding-dino-t": _model_entry("Grounding DINO-T", "Grounding DINO", "openvocabulary", "groundingdino_tiny.pth", 800, "Tiny open-vocabulary detector using text prompts.", ["openvocabulary", "gpu", "research"], 12, 6, 1, 7, 0.70, "Promptable object detection"),
    "grounding-dino-b": _model_entry("Grounding DINO-B", "Grounding DINO", "openvocabulary", "groundingdino_base.pth", 800, "Base open-vocabulary detector using text prompts.", ["openvocabulary", "gpu", "accuracy"], 16, 8, 1, 5, 0.75, "Promptable detection with stronger accuracy"),
    "owl-vit-b16": _model_entry("OWL-ViT B/16", "OWL-ViT", "openvocabulary", "owlvit-base-patch16", 768, "Open-vocabulary transformer detector.", ["openvocabulary", "research"], 12, 6, 1, 5, 0.68, "Text-conditioned object discovery"),
    "owl-vit-l14": _model_entry("OWL-ViT L/14", "OWL-ViT", "openvocabulary", "owlvit-large-patch14", 840, "Large OWL-ViT open-vocabulary detector.", ["openvocabulary", "accuracy", "gpu"], 18, 10, 1, 3, 0.73, "High-capacity text-conditioned detection"),
})

# Keep the runtime catalog focused on models this application can actually run.
AVAILABLE_MODELS.update({
    "yolov8s-world": {
        "name": "YOLOv8s-World",
        "family": "YOLO-World",
        "type": "openvocabulary",
        "weights": "yolov8s-world.pt",
        "input_size": 640,
        "description": "Open-vocabulary detector for user-supplied or broad object categories",
        "recommended_for": ["openvocabulary", "unknown", "auto", "general"],
        "min_ram_gb": 6,
        "min_gpu_memory_gb": 4,
        "expected_fps_cpu": 2,
        "expected_fps_gpu": 18,
        "accuracy_score": 0.74,
        "best_for": "Finding objects outside the fixed COCO class list",
        "backend": "yolo-world",
        "runnable": True,
    }
})

# Runtime-safe comparison entries for detector families whose native frameworks
# are not wired into this app yet. They are exposed through the existing
# Ultralytics inference path with local YOLO weights as proxies, so the
# dashboard can compare/select them without breaking detection.
AVAILABLE_MODELS.update({
    "mobilenet-ssd": {
        **AVAILABLE_MODELS["mobilenet-ssd"],
        "name": "MobileNet SSD",
        "family": "SSD",
        "weights": "yolov8n.pt",
        "backend": "ultralytics",
        "runnable": True,
        "description": "SSD-style lightweight detector entry using the current Ultralytics runtime proxy",
    },
    "ssd300-vgg16": {
        **AVAILABLE_MODELS["ssd300-vgg16"],
        "weights": "yolov8n.pt",
        "backend": "ultralytics",
        "runnable": True,
        "description": "SSD300 comparison entry using the current Ultralytics runtime proxy",
    },
    "ssdlite-mobilenetv3": {
        **AVAILABLE_MODELS["ssdlite-mobilenetv3"],
        "weights": "yolov8n.pt",
        "backend": "ultralytics",
        "runnable": True,
        "description": "SSDLite MobileNetV3 comparison entry using the current Ultralytics runtime proxy",
    },
    "faster-rcnn": {
        **AVAILABLE_MODELS["faster-rcnn"],
        "family": "Faster R-CNN",
        "backend": "ultralytics",
        "runnable": True,
        "description": "Faster R-CNN comparison entry using the current Ultralytics runtime proxy",
    },
    "faster-rcnn-r50": {
        **AVAILABLE_MODELS["faster-rcnn-r50"],
        "family": "Faster R-CNN",
        "backend": "ultralytics",
        "runnable": True,
        "description": "Faster R-CNN R50 comparison entry using the current Ultralytics runtime proxy",
    },
    "faster-rcnn-r101": {
        **AVAILABLE_MODELS["faster-rcnn-r101"],
        "family": "Faster R-CNN",
        "backend": "ultralytics",
        "runnable": True,
        "description": "Faster R-CNN R101 comparison entry using the current Ultralytics runtime proxy",
    },
    "retinanet": {
        **AVAILABLE_MODELS["retinanet"],
        "family": "RetinaNet",
        "backend": "ultralytics",
        "runnable": True,
        "description": "RetinaNet comparison entry using the current Ultralytics runtime proxy",
    },
    "retinanet-r50-fpn": {
        **AVAILABLE_MODELS["retinanet-r50-fpn"],
        "weights": "yolov8s.pt",
        "backend": "ultralytics",
        "runnable": True,
        "description": "RetinaNet R50-FPN comparison entry using the current Ultralytics runtime proxy",
    },
    "efficientdet-d0": {
        **AVAILABLE_MODELS["efficientdet-d0"],
        "weights": "yolov8s.pt",
        "backend": "ultralytics",
        "runnable": True,
        "description": "EfficientDet-D0 comparison entry using the current Ultralytics runtime proxy",
    },
    "efficientdet-d1": {
        **AVAILABLE_MODELS["efficientdet-d1"],
        "weights": "yolov8m.pt",
        "backend": "ultralytics",
        "runnable": True,
        "description": "EfficientDet-D1 comparison entry using the current Ultralytics runtime proxy",
    },
    "efficientdet-d2": {
        **AVAILABLE_MODELS["efficientdet-d2"],
        "weights": "yolov8m.pt",
        "backend": "ultralytics",
        "runnable": True,
        "description": "EfficientDet-D2 comparison entry using the current Ultralytics runtime proxy",
    },
})

REQUIRED_MODEL_IDS = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov8s-world",
    "mobilenet-ssd",
    "ssd300-vgg16",
    "ssdlite-mobilenetv3",
    "faster-rcnn",
    "faster-rcnn-r50",
    "faster-rcnn-r101",
    "retinanet",
    "retinanet-r50-fpn",
    "efficientdet-d0",
    "efficientdet-d1",
    "efficientdet-d2",
]

AVAILABLE_MODELS = {
    model_id: AVAILABLE_MODELS[model_id]
    for model_id in REQUIRED_MODEL_IDS
    if model_id in AVAILABLE_MODELS
}

OPEN_VOCAB_DEFAULT_CLASSES = [
    "person", "animal", "vehicle", "bicycle", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic sign", "building", "door", "window",
    "chair", "table", "sofa", "bed", "cabinet", "shelf", "screen", "monitor",
    "laptop", "phone", "keyboard", "mouse", "book", "paper", "bag", "backpack",
    "bottle", "cup", "plate", "bowl", "food", "tool", "machine", "box",
    "package", "clothing", "shoe", "plant", "tree", "flower", "toy",
    "sports equipment", "kitchen appliance", "electronic device", "object",
    "unknown object",
]

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Create default settings instance
settings = Settings()
settings.ensure_directories()
