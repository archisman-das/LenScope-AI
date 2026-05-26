"""
GradCAM explainability for detection models.

The service uses real activation/gradient hooks when the backend exposes a
PyTorch module. YOLO-family models use the penultimate module in the
Ultralytics detection graph; torchvision-style detectors use backbone.layer4.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..config import settings
from ..utils.image import load_image, save_image
from .detection import DetectionService

logger = logging.getLogger(__name__)


class GradCAMService:
    """Generate GradCAM heatmaps and overlay explanations."""

    def __init__(self) -> None:
        self.detection_service = DetectionService()

    def explain(
        self,
        image_source: Any,
        model_id: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        image = load_image(image_source)
        engine = self.detection_service.get_engine(model_id)
        detections, inference_time_ms = engine.detect(
            image,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=100,
        )

        heatmap = self.compute_heatmap(engine.model, image)
        overlay, overlay_path = self.save_overlay(image, heatmap)
        per_class_activation = self.per_class_activation_map(heatmap, detections, image.shape[:2])

        return {
            "model_id": model_id,
            "detections": detections,
            "inference_time_ms": round(inference_time_ms, 2),
            "heatmap": heatmap,
            "overlay_image": overlay,
            "gradcam_overlay_path": str(overlay_path),
            "gradcam_overlay_url": f"/uploads/{overlay_path.name}",
            "per_class_activation_map": per_class_activation,
        }

    def compute_heatmap(self, model_wrapper: Any, image: np.ndarray) -> np.ndarray:
        module = self._unwrap_torch_module(model_wrapper)
        target_layer = self._target_layer(model_wrapper, module)
        if module is None or target_layer is None:
            logger.warning("No hookable PyTorch layer found for GradCAM; using edge fallback")
            return self._fallback_heatmap(image)

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(_module, _inputs, output):
            activations.append(output if torch.is_tensor(output) else output[0])

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            module.zero_grad(set_to_none=True)
            tensor = self._preprocess_for_model(image, next(module.parameters()).device)
            output = module(tensor)
            target = self._select_target_score(output)
            if target is None:
                return self._fallback_heatmap(image)
            target.backward(retain_graph=False)

            if not activations or not gradients:
                return self._fallback_heatmap(image)

            activation = activations[-1].detach()
            gradient = gradients[-1].detach()
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1))[0]
            cam = cam.cpu().numpy()
            return self._normalize_and_resize(cam, image.shape[:2])
        except Exception as exc:
            logger.warning("GradCAM computation failed: %s", exc)
            return self._fallback_heatmap(image)
        finally:
            forward_handle.remove()
            backward_handle.remove()

    def save_overlay(self, image: np.ndarray, heatmap: np.ndarray) -> Tuple[np.ndarray, Path]:
        heatmap_u8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
        color_heatmap = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, color_heatmap, 0.4, 0)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        output_path = settings.upload_path / f"gradcam_{timestamp}.jpg"
        save_image(overlay, str(output_path))
        return overlay, output_path

    def per_class_activation_map(
        self,
        heatmap: np.ndarray,
        detections: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        height, width = image_shape
        by_class: Dict[str, List[float]] = {}
        regions_by_class: Dict[str, List[str]] = {}

        for detection in detections:
            class_name = detection["class_name"]
            bbox = detection["bbox"]
            x1 = max(0, min(width - 1, int(bbox["x1"])))
            y1 = max(0, min(height - 1, int(bbox["y1"])))
            x2 = max(x1 + 1, min(width, int(bbox["x2"])))
            y2 = max(y1 + 1, min(height, int(bbox["y2"])))
            region = heatmap[y1:y2, x1:x2]
            activation = float(region.mean()) if region.size else 0.0
            by_class.setdefault(class_name, []).append(activation)
            regions_by_class.setdefault(class_name, []).append(self._region_name(x1, y1, x2, y2, width, height))

        return {
            class_name: {
                "mean_activation": round(float(np.mean(values)), 4),
                "max_activation": round(float(np.max(values)), 4),
                "primary_region": self._most_common(regions_by_class[class_name]),
                "regions": regions_by_class[class_name],
            }
            for class_name, values in by_class.items()
        }

    def enrich_detections_with_activation(
        self,
        detections: List[Dict[str, Any]],
        activation_map: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        enriched = []
        for detection in detections:
            item = dict(detection)
            class_activation = activation_map.get(item["class_name"])
            item["metadata"] = {
                **item.get("metadata", {}),
                "gradcam_activation": class_activation,
            }
            enriched.append(item)
        return enriched

    @staticmethod
    def _unwrap_torch_module(model_wrapper: Any) -> Optional[torch.nn.Module]:
        if isinstance(model_wrapper, torch.nn.Module):
            return model_wrapper
        module = getattr(model_wrapper, "model", None)
        if isinstance(module, torch.nn.Module):
            return module
        return None

    @staticmethod
    def _target_layer(model_wrapper: Any, module: Optional[torch.nn.Module]) -> Optional[torch.nn.Module]:
        if module is None:
            return None

        yolo_model = getattr(model_wrapper, "model", None)
        yolo_layers = getattr(yolo_model, "model", None)
        if yolo_layers is not None and len(yolo_layers) >= 2:
            return yolo_layers[-2]

        backbone = getattr(module, "backbone", None)
        layer4 = getattr(backbone, "layer4", None)
        if layer4 is not None:
            return layer4

        layer4 = getattr(module, "layer4", None)
        if layer4 is not None:
            return layer4

        conv_layers = [layer for layer in module.modules() if isinstance(layer, torch.nn.Conv2d)]
        return conv_layers[-1] if conv_layers else None

    @staticmethod
    def _preprocess_for_model(image: np.ndarray, device: torch.device) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return tensor.to(device)

    @staticmethod
    def _select_target_score(output: Any) -> Optional[torch.Tensor]:
        tensors = []
        if torch.is_tensor(output):
            tensors = [output]
        elif isinstance(output, (list, tuple)):
            tensors = [item for item in output if torch.is_tensor(item)]

        candidates = []
        for tensor in tensors:
            if tensor.requires_grad and tensor.numel() > 0:
                candidates.append(tensor.reshape(-1).max())
        if not candidates:
            return None
        return torch.stack(candidates).max()

    @staticmethod
    def _normalize_and_resize(cam: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value
        return cv2.resize(cam, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    @staticmethod
    def _fallback_heatmap(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        edges = np.abs(edges)
        if edges.max() > 0:
            edges = edges / edges.max()
        return cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 9)

    @staticmethod
    def _region_name(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> str:
        center_x = ((x1 + x2) / 2) / max(width, 1)
        center_y = ((y1 + y2) / 2) / max(height, 1)
        vertical = "upper" if center_y < 0.33 else "lower" if center_y > 0.66 else "middle"
        horizontal = "left" if center_x < 0.33 else "right" if center_x > 0.66 else "center"
        return f"{vertical}-{horizontal}"

    @staticmethod
    def _most_common(values: List[str]) -> Optional[str]:
        if not values:
            return None
        return max(set(values), key=values.count)


gradcam_service = GradCAMService()
