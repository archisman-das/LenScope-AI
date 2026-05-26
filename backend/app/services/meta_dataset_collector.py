"""
Non-blocking metadata collection for training the model router.

The collector intentionally does expensive work in a single background worker:
image statistics, a fast YOLOv8n object-count pass, and CLIP scene labeling.
Detection requests only enqueue a small payload and continue.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..config import settings
from ..utils.image import load_image

logger = logging.getLogger(__name__)


SCENE_LABELS = ["indoor", "outdoor", "crowd", "vehicle", "night"]


class MetaDatasetCollector:
    """Collect scene/model performance rows without blocking inference."""

    _instance: Optional["MetaDatasetCollector"] = None

    def __new__(cls) -> "MetaDatasetCollector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.output_path = settings.project_root / "backend" / "data" / "meta_training_data.jsonl"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="meta-dataset")
        self._pending_slots = threading.BoundedSemaphore(value=16)
        self._write_lock = threading.Lock()
        self._models_lock = threading.Lock()
        self._yolo_model = None
        self._clip_model = None
        self._clip_processor = None
        self._clip_device = None
        self._initialized = True

    def collect(
        self,
        image_path: Any,
        model_id: str,
        map_score: Optional[float],
        latency_ms: Optional[float],
    ) -> bool:
        """
        Queue a training row for the model router.

        Args:
            image_path: Image file path or already-loaded image array.
            model_id: Detection model used for the primary inference.
            map_score: mAP achieved by that model, if known.
            latency_ms: Primary inference latency in milliseconds.

        Returns:
            True when the row was queued, False when the queue is saturated.
        """
        if not self._pending_slots.acquire(blocking=False):
            logger.debug("Skipping meta dataset collection because the queue is full")
            return False

        image_source = image_path.copy() if isinstance(image_path, np.ndarray) else image_path
        self._executor.submit(
            self._safe_collect,
            image_source,
            model_id,
            map_score,
            latency_ms,
        )
        return True

    def _safe_collect(
        self,
        image_source: Any,
        model_id: str,
        map_score: Optional[float],
        latency_ms: Optional[float],
    ) -> None:
        try:
            image = load_image(image_source)
            scene_features = self._extract_scene_features(image)
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "scene_features": scene_features,
                "model_used": model_id,
                "map_score": map_score,
                "latency_ms": latency_ms,
            }
            self._append_record(record)
        except Exception as exc:
            logger.warning("Meta dataset collection failed: %s", exc)
        finally:
            self._pending_slots.release()

    def _extract_scene_features(self, image: np.ndarray) -> Dict[str, Any]:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return {
            "mean_brightness": round(float(np.mean(gray)), 4),
            "contrast": round(float(np.std(gray)), 4),
            "blur_score": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
            "estimated_object_count": self._estimate_object_count(image),
            "aspect_ratio": round(float(width / height), 4) if height else 0,
            "width": int(width),
            "height": int(height),
            "resolution_bucket": self._resolution_bucket(width, height),
            "scene_type": self._classify_scene_type(image),
        }

    def _estimate_object_count(self, image: np.ndarray) -> Optional[int]:
        try:
            model = self._get_yolo_model()
            results = model(image, conf=0.25, iou=0.45, max_det=300, verbose=False)
            if not results:
                return 0
            boxes = getattr(results[0], "boxes", None)
            return int(len(boxes)) if boxes is not None else 0
        except Exception as exc:
            logger.debug("YOLOv8n object-count pass failed: %s", exc)
            return None

    def _classify_scene_type(self, image: np.ndarray) -> Optional[str]:
        try:
            import torch
            from PIL import Image

            model, processor, device = self._get_clip()
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            prompts = [f"a photo of a {label} scene" for label in SCENE_LABELS]
            inputs = processor(text=prompts, images=pil_image, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]

            return SCENE_LABELS[int(probs.argmax().item())]
        except Exception as exc:
            logger.debug("CLIP scene classification failed: %s", exc)
            return None

    def _get_yolo_model(self) -> Any:
        with self._models_lock:
            if self._yolo_model is None:
                from .model_manager import ModelManager

                self._yolo_model = ModelManager().load_model("yolov8n")
            return self._yolo_model

    def _get_clip(self) -> tuple[Any, Any, Any]:
        with self._models_lock:
            if self._clip_model is None or self._clip_processor is None:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                device = "cuda" if torch.cuda.is_available() and settings.DEVICE != "cpu" else "cpu"
                self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self._clip_model.to(device)
                self._clip_model.eval()
                self._clip_device = device

            return self._clip_model, self._clip_processor, self._clip_device

    def _append_record(self, record: Dict[str, Any]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            with self.output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _resolution_bucket(width: int, height: int) -> str:
        pixels = width * height
        if pixels < 640 * 480:
            return "small"
        if pixels <= 1920 * 1080:
            return "medium"
        return "large"


meta_dataset_collector = MetaDatasetCollector()
