"""
Detection Service - Core AI inference engine for object detection
Handles image processing, model inference, and result formatting
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

from ..config import settings, AVAILABLE_MODELS, COCO_CLASSES, OPEN_VOCAB_DEFAULT_CLASSES
from ..utils.image import load_image, draw_detections, save_image, encode_image_base64, resize_image
from ..utils.helpers import Timer

logger = logging.getLogger(__name__)


class DetectionEngine:
    """
    Core detection engine that wraps the YOLO model
    Handles inference with various optimization options
    """
    
    def __init__(self, model_id: str = None):
        """
        Initialize detection engine
        
        Args:
            model_id: Model identifier (e.g., 'yolov8n')
        """
        self.model_id = model_id or settings.DEFAULT_MODEL
        self.model = None
        self.device = None
        self._initialized = False
    
    def initialize(self, model_manager) -> 'DetectionEngine':
        """
        Initialize the model using ModelManager
        
        Args:
            model_manager: ModelManager instance
            
        Returns:
            Self for chaining
        """
        from .model_manager import ModelManager
        
        if self.model is None:
            self.model_manager = model_manager or ModelManager()
            self.model = self.model_manager.load_model(self.model_id)
            self.device = self.model_manager.get_device()
            self._initialized = True
            logger.info(f"Detection engine initialized with model: {self.model_id}")
        
        return self
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        max_detections: int = None,
        use_onnx: bool = False,
        use_fp16: bool = False,
        open_vocab_classes: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Run detection on an image
        
        Args:
            image: Input image (BGR numpy array)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: NMS IoU threshold
            max_detections: Maximum number of detections
            use_onnx: Use ONNX runtime for inference
            use_fp16: Use FP16 precision (requires GPU support)
        
        Returns:
            (detections, inference_time_ms)
        """
        if not self._initialized:
            raise RuntimeError("Detection engine not initialized. Call initialize() first.")
        
        # Validate input image
        if image is None or not isinstance(image, np.ndarray):
            logger.warning("Invalid image input to detection engine")
            return [], 0.0
        
        if image.size == 0:
            logger.warning("Empty image array provided to detection engine")
            return [], 0.0
        
        # Ensure image has valid dimensions
        if len(image.shape) < 2 or image.shape[0] == 0 or image.shape[1] == 0:
            logger.warning(f"Image has invalid dimensions: {image.shape}")
            return [], 0.0
        
        conf_thresh = confidence_threshold or settings.CONFIDENCE_THRESHOLD
        iou_thresh = iou_threshold or settings.IOU_THRESHOLD
        max_det = max_detections or settings.MAX_DETECTIONS
        
        timer = Timer()
        timer.start()
        
        try:
            if AVAILABLE_MODELS.get(self.model_id, {}).get("backend") == "yolo-world":
                classes = open_vocab_classes or OPEN_VOCAB_DEFAULT_CLASSES
                self.model.set_classes(classes)
            
            # Run inference
            results = self.model(
                image,
                conf=conf_thresh,
                iou=iou_thresh,
                max_det=max_det,
                verbose=False
            )
            
            timer.stop()
            inference_time_ms = timer.elapsed_ms()
            
            # Process results
            if results is None or len(results) == 0:
                logger.warning("Model returned empty results")
                return [], inference_time_ms
            
            detections = self._process_results(results[0])
            
            return detections, inference_time_ms
            
        except Exception as e:
            timer.stop()
            logger.error(f"Detection failed: {e}")
            # Return empty results instead of raising exception for graceful degradation
            return [], timer.elapsed_ms() if timer.elapsed_ms() > 0 else 0.0
    
    def _process_results(self, result) -> List[Dict[str, Any]]:
        """
        Process YOLO results into standardized format
        
        Args:
            result: YOLO result object
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes
        
        for i in range(len(boxes)):
            # Get box coordinates
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box.tolist()
            
            # Get class and confidence
            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())
            
            # Get class name from the model result first. This is required for
            # open-vocabulary models whose labels are not COCO ids.
            result_names = getattr(result, "names", None) or {}
            if isinstance(result_names, dict) and class_id in result_names:
                class_name = result_names[class_id]
            elif isinstance(result_names, list) and class_id < len(result_names):
                class_name = result_names[class_id]
            else:
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            }
            
            detections.append(detection)
        
        return detections

    def supports_open_vocabulary(self) -> bool:
        return AVAILABLE_MODELS.get(self.model_id, {}).get("backend") == "yolo-world"


class DetectionService:
    """
    High-level detection service that handles the full detection workflow
    Including image loading, detection, visualization, and result storage
    """
    
    def __init__(self):
        """Initialize detection service"""
        self.engines: Dict[str, DetectionEngine] = {}
        self.model_manager = None
    
    def get_engine(self, model_id: str) -> DetectionEngine:
        """
        Get or create a detection engine for a model
        
        Args:
            model_id: Model identifier
            
        Returns:
            DetectionEngine instance
        """
        if model_id not in self.engines:
            from .model_manager import ModelManager
            
            self.model_manager = ModelManager()
            engine = DetectionEngine(model_id)
            engine.initialize(self.model_manager)
            self.engines[model_id] = engine
        
        return self.engines[model_id]
    
    def detect_image(
        self,
        image_source: Any,
        model_id: str,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        draw_boxes: bool = True,
        return_base64: bool = False,
        save_path: str = None,
        open_vocab_classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Full detection workflow for an image
        
        Args:
            image_source: Image file path, URL, bytes, or numpy array
            model_id: Model identifier
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            draw_boxes: Whether to draw bounding boxes on output
            return_base64: Whether to return image as base64
            save_path: Path to save output image
            
        Returns:
            Dictionary with detection results and metadata
        """
        # Load image
        image = load_image(image_source)
        original_height, original_width = image.shape[:2]  # (height, width)
        
        # Get detection engine
        engine = self.get_engine(model_id)
        
        # Run detection
        total_start = time.time()
        detections, inference_time_ms = engine.detect(
            image,
            confidence_threshold,
            iou_threshold,
            open_vocab_classes=open_vocab_classes
        )
        total_time = (time.time() - total_start) * 1000
        
        # Prepare result
        result = {
            "model_id": model_id,
            "original_width": original_width,
            "original_height": original_height,
            "num_detections": len(detections),
            "inference_time_ms": round(inference_time_ms, 2),
            "total_time_ms": round(total_time, 2),
            "fps": round(1000 / inference_time_ms, 1) if inference_time_ms > 0 else 0,
            "detections": detections,
            "class_summary": self._get_class_summary(detections),
            "device_used": str(engine.device)
        }
        
        # Draw detections on image if requested
        if draw_boxes and detections:
            output_image = draw_detections(image, detections)
            
            # Add FPS and inference time overlay
            self._add_overlay(output_image, result)
            
            if return_base64:
                result["output_image_base64"] = encode_image_base64(output_image)
            
            if save_path:
                save_image(output_image, save_path)
                result["output_path"] = save_path
        else:
            if return_base64:
                result["output_image_base64"] = encode_image_base64(image)
        
        return result

    def detect_image_best(
        self,
        image_source: Any,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        draw_boxes: bool = True,
        return_base64: bool = False,
        save_path: str = None,
        open_vocab_classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Automatically choose the best available detector for an uploaded image.

        Uses YOLO-World first when available because it can detect labels
        outside the fixed COCO list. Falls back to a standard COCO detector if
        open-vocabulary inference cannot find anything.
        """
        image = load_image(image_source)
        standard_model = self.auto_select_model(image)

        if "yolov8s-world" in AVAILABLE_MODELS:
            open_result = self.detect_image(
                image_source=image,
                model_id="yolov8s-world",
                confidence_threshold=max(confidence_threshold or 0.25, 0.15),
                iou_threshold=iou_threshold,
                draw_boxes=draw_boxes,
                return_base64=return_base64,
                save_path=save_path,
                open_vocab_classes=open_vocab_classes or OPEN_VOCAB_DEFAULT_CLASSES,
            )
            open_result["auto_selection"] = {
                "strategy": "open-vocabulary-first",
                "selected_model": "yolov8s-world",
                "fallback_used": False,
                "reason": "open-vocabulary detector selected for broad object coverage",
            }

            if open_result["num_detections"] > 0:
                return open_result

        standard_result = self.detect_image(
            image_source=image,
            model_id=standard_model,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            draw_boxes=draw_boxes,
            return_base64=return_base64,
            save_path=save_path,
        )
        standard_result["auto_selection"] = {
            "strategy": "open-vocabulary-first",
            "selected_model": standard_model,
            "fallback_used": "yolov8s-world" in AVAILABLE_MODELS,
            "reason": "open-vocabulary detector found nothing, so standard detection was used",
        }

        return standard_result
    
    def compare_models(
        self,
        image_source: Any,
        model_ids: List[str],
        confidence_threshold: float = None,
        iou_threshold: float = None,
        open_vocab_classes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same image
        
        Args:
            image_source: Image source
            model_ids: List of model identifiers
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            
        Returns:
            Comparison results
        """
        comparison = {
            "models": {},
            "summary": {}
        }
        
        total_detections = {}
        inference_times = {}
        
        for model_id in model_ids:
            try:
                result = self.detect_image(
                    image_source,
                    model_id,
                    confidence_threshold,
                    iou_threshold,
                    draw_boxes=False,
                    open_vocab_classes=open_vocab_classes
                )
                
                comparison["models"][model_id] = result
                total_detections[model_id] = result["num_detections"]
                inference_times[model_id] = result["inference_time_ms"]
                
            except Exception as e:
                comparison["models"][model_id] = {"error": str(e)}
        
        # Find best model for different criteria
        if total_detections:
            comparison["summary"] = {
                "fastest_model": min(inference_times, key=inference_times.get) if inference_times else None,
                "most_detections": max(total_detections, key=total_detections.get) if total_detections else None,
                "recommendation": self._get_recommendation(model_ids, inference_times, total_detections)
            }
        
        return comparison
    
    def _get_class_summary(self, detections: List[Dict]) -> Dict[str, Any]:
        """Get summary of detected classes"""
        class_counts = {}
        class_confidences = {}
        
        for det in detections:
            class_name = det["class_name"]
            if class_name not in class_counts:
                class_counts[class_name] = 0
                class_confidences[class_name] = []
            
            class_counts[class_name] += 1
            class_confidences[class_name].append(det["confidence"])
        
        # Calculate average confidence per class
        avg_confidences = {
            k: round(sum(v) / len(v), 3) 
            for k, v in class_confidences.items()
        }
        
        return {
            "counts": class_counts,
            "avg_confidence": avg_confidences,
            "total_classes": len(class_counts)
        }
    
    def _add_overlay(self, image: np.ndarray, result: Dict) -> None:
        """Add FPS and timing overlay to image"""
        import cv2
        
        # Prepare text
        fps_text = f"FPS: {result['fps']}"
        time_text = f"Inference: {result['inference_time_ms']:.1f}ms"
        det_text = f"Detections: {result['num_detections']}"
        model_text = f"Model: {result['model_id']}"
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (350, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 255)  # Cyan
        
        cv2.putText(image, model_text, (10, 25), font, font_scale, color, 1)
        cv2.putText(image, fps_text, (10, 45), font, font_scale, color, 1)
        cv2.putText(image, time_text, (10, 65), font, font_scale, color, 1)
        cv2.putText(image, det_text, (10, 85), font, font_scale, color, 1)
    
    def _get_recommendation(
        self,
        model_ids: List[str],
        inference_times: Dict[str, float],
        total_detections: Dict[str, int]
    ) -> str:
        """Get model recommendation based on results"""
        if not inference_times:
            return "No valid results"
        
        fastest = min(inference_times, key=inference_times.get)
        fastest_time = inference_times[fastest]
        
        # Recommendation logic
        if fastest_time < 30:
            return f"{fastest} recommended for real-time applications ({fastest_time:.1f}ms)"
        elif fastest_time < 100:
            return f"{fastest} offers good balance of speed and accuracy ({fastest_time:.1f}ms)"
        else:
            return f"Consider smaller models for faster inference. {fastest} is fastest at {fastest_time:.1f}ms"
    
    def get_model_recommendation(self, use_case: str = "realtime") -> str:
        """
        Get model recommendation for a use case
        
        Args:
            use_case: 'realtime', 'accuracy', 'balanced'
            
        Returns:
            Recommended model ID
        """
        from .model_manager import ModelManager
        
        manager = ModelManager()
        return manager.recommend_model(use_case)
    
    def auto_select_model(self, image: np.ndarray, use_case: str = "auto") -> str:
        """
        Automatically select the best model based on image analysis
        
        Args:
            image: Input image (numpy array)
            use_case: 'auto', 'realtime', 'accuracy', 'balanced'
            
        Returns:
            Recommended model ID
        """
        height, width = image.shape[:2]
        resolution = height * width
        
        # Analyze image characteristics
        is_high_resolution = resolution > 1920 * 1080  # > 2MP
        is_low_resolution = resolution < 640 * 480     # < 0.3MP
        
        # Calculate image complexity (simple heuristic based on edge density)
        try:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            is_complex_scene = edge_density > 0.15  # High edge density = complex scene
            is_simple_scene = edge_density < 0.05   # Low edge density = simple scene
        except:
            is_complex_scene = False
            is_simple_scene = False
        
        # Determine best model based on analysis
        if "yolov8s-world" in AVAILABLE_MODELS and use_case in {"unknown", "openvocabulary"}:
            return "yolov8s-world"

        if use_case == "realtime" or (use_case == "auto" and not is_high_resolution):
            # For real-time or low-res images, prefer fast models
            if is_simple_scene:
                return "yolov8n"  # Simple scene, use fastest
            else:
                return "yolov8s"  # Some complexity, use small
        
        elif use_case == "accuracy" or (use_case == "auto" and is_high_resolution and is_complex_scene):
            # For high-res complex images, prefer accurate models
            if is_high_resolution:
                return "yolov8l"  # High-res, use large model
            else:
                return "yolov8m"  # Complex but not huge, use medium
        
        else:  # balanced or auto
            if is_high_resolution:
                return "yolov8m"  # High-res balanced
            elif is_complex_scene:
                return "yolov8s"  # Complex scene balanced
            else:
                return "yolov8n"  # Simple scene balanced
    
    def detect_image_auto(
        self,
        image_source: Any,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        draw_boxes: bool = True,
        return_base64: bool = False,
        save_path: str = None,
        auto_model: bool = True
    ) -> Dict[str, Any]:
        """
        Full detection workflow with automatic model selection
        
        Args:
            image_source: Image file path, URL, bytes, or numpy array
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            draw_boxes: Whether to draw bounding boxes on output
            return_base64: Whether to return image as base64
            save_path: Path to save output image
            auto_model: Whether to use automatic model selection
            
        Returns:
            Dictionary with detection results and metadata
        """
        # Load image
        image = load_image(image_source)
        original_height, original_width = image.shape[:2]
        
        # Auto-select model if requested
        if auto_model:
            model_id = self.auto_select_model(image)
            logger.info(f"Auto-selected model: {model_id} for image {original_width}x{original_height}")
        else:
            model_id = settings.DEFAULT_MODEL
        
        # Run detection with selected model
        return self.detect_image(
            image_source=image,
            model_id=model_id,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            draw_boxes=draw_boxes,
            return_base64=return_base64,
            save_path=save_path
        )
