"""
Adaptive AI Model Switching Service
Intelligently switches between AI models based on:
- Scene complexity
- Hardware resources (CPU/GPU availability, RAM, VRAM)
- Performance requirements
- Detection history
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np
import psutil

from ..config import AVAILABLE_MODELS, COCO_CLASSES

logger = logging.getLogger(__name__)


class SceneComplexityAnalyzer:
    """Analyzes scene complexity to determine optimal model"""
    
    @staticmethod
    def analyze(image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image complexity metrics
        
        Args:
            image: Input image (BGR numpy array)
            
        Returns:
            Dictionary with complexity metrics
        """
        try:
            import cv2
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate edge density using Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0)) / edges.size
            
            # Calculate texture complexity using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_complexity = float(np.var(laplacian))
            
            # Calculate brightness distribution
            brightness_std = float(np.std(gray))
            brightness_mean = float(np.mean(gray))
            
            # Estimate scene depth (simple heuristic based on edge distribution)
            height, width = gray.shape
            top_edges = float(np.sum(edges[:height//3] > 0))
            bottom_edges = float(np.sum(edges[2*height//3:] > 0))
            depth_indicator = abs(top_edges - bottom_edges) / max(top_edges, bottom_edges, 1)
            
            # Calculate object density estimate (based on edge clustering)
            kernel = np.ones((5, 5), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)
            eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
            object_density = float(np.sum(eroded_edges > 0)) / eroded_edges.size
            
            # Overall complexity score (0-100)
            complexity_score = (
                edge_density * 40 +
                min(texture_complexity / 1000, 1) * 30 +
                object_density * 30
            )
            
            return {
                "edge_density": round(edge_density, 4),
                "texture_complexity": round(texture_complexity, 2),
                "brightness_std": round(brightness_std, 2),
                "brightness_mean": round(brightness_mean, 2),
                "depth_indicator": round(depth_indicator, 4),
                "object_density": round(object_density, 4),
                "complexity_score": round(complexity_score, 2),
                "complexity_level": SceneComplexityAnalyzer._get_complexity_level(complexity_score)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing scene complexity: {e}")
            return {
                "edge_density": 0,
                "texture_complexity": 0,
                "brightness_std": 0,
                "brightness_mean": 0,
                "depth_indicator": 0,
                "object_density": 0,
                "complexity_score": 50,
                "complexity_level": "medium"
            }
    
    @staticmethod
    def _get_complexity_level(score: float) -> str:
        """Convert complexity score to level"""
        if score < 20:
            return "very_low"
        elif score < 40:
            return "low"
        elif score < 60:
            return "medium"
        elif score < 80:
            return "high"
        else:
            return "very_high"


class AdaptiveModelSwitcher:
    """
    Intelligently switches between AI models based on:
    - Scene complexity
    - Performance requirements
    - Historical performance data
    - Resource availability
    """
    
    def __init__(self):
        """Initialize adaptive model switcher"""
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.scene_history: List[Dict[str, Any]] = []
        self.model_switches: List[Dict[str, Any]] = []
        self.complexity_analyzer = SceneComplexityAnalyzer()
        
        # Model performance profiles (will be updated with real data)
        self.model_profiles = self._initialize_model_profiles()
        
        # Current state
        self.current_model: Optional[str] = None
        self.last_switch_time: Optional[datetime] = None
        self.switch_cooldown: float = 5.0  # Minimum seconds between switches
        
    def _initialize_model_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize model performance profiles"""
        profiles = {}
        for model_id, config in AVAILABLE_MODELS.items():
            profiles[model_id] = {
                "model_id": model_id,
                "name": config["name"],
                "type": config["type"],
                "input_size": config["input_size"],
                "recommended_for": config["recommended_for"],
                "avg_inference_time": 0,
                "avg_accuracy_score": 0,
                "total_detections": 0,
                "success_rate": 1.0,
                "last_used": None,
                "complexity_preference": self._get_complexity_preference(config["type"])
            }
        return profiles
    
    def _get_complexity_preference(self, model_type: str) -> str:
        """Get preferred complexity level for model type"""
        if model_type == "realtime":
            return "low"
        elif model_type == "research":
            return "high"
        elif model_type == "openvocabulary":
            return "medium"
        return "medium"
    
    def analyze_and_select(
        self, 
        image: np.ndarray, 
        use_case: str = "auto",
        current_performance: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze scene and select optimal model
        
        Args:
            image: Input image
            use_case: Use case type (auto, realtime, accuracy, balanced)
            current_performance: Current system performance metrics
            
        Returns:
            Tuple of (selected_model_id, analysis_info)
        """
        # Analyze scene complexity
        complexity_info = self.complexity_analyzer.analyze(image)
        
        # Get image dimensions
        height, width = image.shape[:2]
        resolution = height * width
        
        # Determine selection criteria
        selected_model, reasons = self._select_model(
            complexity_info=complexity_info,
            use_case=use_case,
            resolution=resolution,
            current_performance=current_performance
        )
        
        # Record scene analysis
        scene_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "image_dimensions": {"width": width, "height": height},
            "resolution": resolution,
            "complexity_info": complexity_info,
            "use_case": use_case,
            "selected_model": selected_model,
            "selection_reasons": reasons
        }
        self.scene_history.append(scene_record)
        
        # Check if we should switch models
        should_switch = self._should_switch_model(selected_model)
        
        if should_switch:
            self._record_switch(selected_model, reasons)
            self.current_model = selected_model
            self.last_switch_time = datetime.utcnow()
        
        analysis_info = {
            "scene_complexity": complexity_info,
            "selected_model": selected_model,
            "should_switch": should_switch,
            "current_model": self.current_model,
            "selection_reasons": reasons,
            "use_case": use_case
        }
        
        return selected_model, analysis_info
    
    def _select_model(
        self,
        complexity_info: Dict[str, Any],
        use_case: str,
        resolution: int,
        current_performance: Optional[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """Select optimal model based on criteria"""
        reasons = []
        candidate_models = [
            model_id
            for model_id, config in AVAILABLE_MODELS.items()
            if config.get("runnable", True)
        ]
        scores = {}
        
        complexity_score = complexity_info["complexity_score"]
        complexity_level = complexity_info["complexity_level"]
        
        for model_id in candidate_models:
            profile = self.model_profiles[model_id]
            score = 0
            
            # Factor 1: Use case matching (40% weight)
            if use_case == "realtime":
                if profile["type"] == "realtime":
                    score += 40
                    reasons.append(f"{model_id}: Optimized for realtime")
                elif "realtime" in profile["recommended_for"]:
                    score += 30
            elif use_case == "accuracy":
                if profile["type"] in ["research", "openvocabulary"]:
                    score += 40
                    reasons.append(f"{model_id}: High accuracy model")
                elif "accuracy" in profile["recommended_for"]:
                    score += 35
            elif use_case == "balanced":
                if "balanced" in profile["recommended_for"]:
                    score += 40
                    reasons.append(f"{model_id}: Balanced performance")
            else:  # auto
                # Auto mode: match complexity with model capability
                if complexity_level in ["very_low", "low"]:
                    if profile["type"] == "realtime":
                        score += 40
                        reasons.append(f"{model_id}: Fast model for simple scene")
                elif complexity_level in ["high", "very_high"]:
                    if profile["type"] in ["research", "openvocabulary"]:
                        score += 40
                        reasons.append(f"{model_id}: Accurate model for complex scene")
                else:
                    if "balanced" in profile["recommended_for"]:
                        score += 35
                        reasons.append(f"{model_id}: Balanced for medium complexity")
            
            # Factor 2: Complexity matching (25% weight)
            model_pref = profile["complexity_preference"]
            complexity_match = {
                "very_low": {"very_low": 25, "low": 20, "medium": 10, "high": 5, "very_high": 0},
                "low": {"very_low": 20, "low": 25, "medium": 15, "high": 10, "very_high": 5},
                "medium": {"very_low": 10, "low": 15, "medium": 25, "high": 15, "very_high": 10},
                "high": {"very_low": 5, "low": 10, "medium": 15, "high": 25, "very_high": 20},
                "very_high": {"very_low": 0, "low": 5, "medium": 10, "high": 20, "very_high": 25}
            }
            score += complexity_match.get(model_pref, {}).get(complexity_level, 10)
            
            # Factor 3: Resolution appropriateness (20% weight)
            model_input_size = profile["input_size"]
            if resolution > 1920 * 1080:  # High resolution
                if model_input_size >= 800:
                    score += 20
                    if f"{model_id}: Suitable for high resolution" not in reasons:
                        reasons.append(f"{model_id}: Suitable for high resolution")
                else:
                    score += 10
            elif resolution < 640 * 480:  # Low resolution
                if model_input_size <= 640:
                    score += 20
                else:
                    score += 10
            else:  # Medium resolution
                if 640 <= model_input_size <= 800:
                    score += 20
                else:
                    score += 15
            
            # Factor 4: Historical performance (15% weight)
            if profile["total_detections"] > 0:
                performance_score = (
                    profile["success_rate"] * 10 +
                    (1 - min(profile["avg_inference_time"] / 1000, 1)) * 5
                )
                score += performance_score
            else:
                score += 7  # Neutral score for new models
            
            scores[model_id] = score
        
        # Select model with highest score
        selected_model = max(scores, key=scores.get)
        
        return selected_model, reasons
    
    def _should_switch_model(self, proposed_model: str) -> bool:
        """Determine if model switch is necessary"""
        if self.current_model is None:
            return True
        
        if proposed_model == self.current_model:
            return False
        
        # Check cooldown
        if self.last_switch_time:
            time_since_last_switch = (datetime.utcnow() - self.last_switch_time).total_seconds()
            if time_since_last_switch < self.switch_cooldown:
                return False
        
        return True
    
    def _record_switch(self, new_model: str, reasons: List[str]):
        """Record model switch event"""
        switch_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_model": self.current_model,
            "to_model": new_model,
            "reasons": reasons
        }
        self.model_switches.append(switch_record)
        
        logger.info(f"Model switched: {self.current_model} -> {new_model}")
    
    def record_performance(
        self, 
        model_id: str, 
        inference_time_ms: float, 
        detection_count: int,
        success: bool = True
    ):
        """Record performance data for a model"""
        profile = self.model_profiles[model_id]
        
        # Update running averages
        total = profile["total_detections"]
        if total > 0:
            profile["avg_inference_time"] = (
                (profile["avg_inference_time"] * total + inference_time_ms) / (total + 1)
            )
            profile["avg_accuracy_score"] = (
                (profile["avg_accuracy_score"] * total + min(detection_count / 10, 1)) / (total + 1)
            )
        else:
            profile["avg_inference_time"] = inference_time_ms
            profile["avg_accuracy_score"] = min(detection_count / 10, 1)
        
        profile["total_detections"] += 1
        profile["success_rate"] = (
            (profile["success_rate"] * total + (1 if success else 0)) / (total + 1)
        )
        profile["last_used"] = datetime.utcnow().isoformat()
        
        # Record in history
        self.performance_history[model_id].append({
            "timestamp": datetime.utcnow().isoformat(),
            "inference_time_ms": inference_time_ms,
            "detection_count": detection_count,
            "success": success
        })
        
        # Keep only last 1000 records per model
        if len(self.performance_history[model_id]) > 1000:
            self.performance_history[model_id] = self.performance_history[model_id][-1000:]
    
    def get_switch_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent model switch history"""
        return self.model_switches[-limit:]
    
    def get_scene_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent scene analysis history"""
        return self.scene_history[-limit:]
    
    def get_model_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all models"""
        stats = {}
        for model_id, profile in self.model_profiles.items():
            if profile["total_detections"] > 0:
                stats[model_id] = {
                    "model_name": profile["name"],
                    "total_detections": profile["total_detections"],
                    "avg_inference_time_ms": round(profile["avg_inference_time"], 2),
                    "avg_accuracy_score": round(profile["avg_accuracy_score"], 3),
                    "success_rate": round(profile["success_rate"], 3),
                    "last_used": profile["last_used"],
                    "complexity_preference": profile["complexity_preference"]
                }
        return stats
    
    def reset_statistics(self):
        """Reset all performance statistics"""
        self.model_profiles = self._initialize_model_profiles()
        self.performance_history.clear()
        self.scene_history.clear()
        self.model_switches.clear()
        self.current_model = None
        self.last_switch_time = None


class HardwareResourceDetector:
    """Detects and monitors hardware resources for model recommendation"""
    
    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """
        Get current system resource information
        
        Returns:
            Dictionary with hardware resource details
        """
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            gpu_info = None
            gpu_memory_total = 0
            gpu_memory_available = 0
            
            if cuda_available:
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    gpu_memory_available = gpu_memory_total - gpu_memory_allocated
                    
                    gpu_info = {
                        "name": gpu_name,
                        "total_memory_gb": round(gpu_memory_total, 2),
                        "allocated_memory_gb": round(gpu_memory_allocated, 2),
                        "reserved_memory_gb": round(gpu_memory_reserved, 2),
                        "available_memory_gb": round(gpu_memory_available, 2)
                    }
                except Exception as e:
                    logger.warning(f"Could not get GPU details: {e}")
            
            # Get system RAM info
            ram = psutil.virtual_memory()
            ram_total_gb = ram.total / (1024**3)
            ram_available_gb = ram.available / (1024**3)
            ram_percent = ram.percent
            
            # Get CPU info
            cpu_count = psutil.cpu_count(logical=True)
            cpu_physical_count = psutil.cpu_count(logical=False) or 1
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Determine resource level
            resource_level = HardwareResourceDetector._classify_resource_level(
                ram_available_gb, gpu_memory_available, cuda_available
            )
            
            return {
                "cpu": {
                    "cores": cpu_count,
                    "physical_cores": cpu_physical_count,
                    "usage_percent": cpu_percent
                },
                "ram": {
                    "total_gb": round(ram_total_gb, 2),
                    "available_gb": round(ram_available_gb, 2),
                    "usage_percent": ram_percent
                },
                "gpu": {
                    "available": cuda_available,
                    "info": gpu_info,
                    "total_memory_gb": gpu_memory_total,
                    "available_memory_gb": round(gpu_memory_available, 2)
                },
                "resource_level": resource_level,
                "recommendation": HardwareResourceDetector._get_resource_recommendation(resource_level, cuda_available)
            }
            
        except Exception as e:
            logger.warning(f"Error detecting hardware resources: {e}")
            return {
                "cpu": {"cores": psutil.cpu_count() or 4, "physical_cores": 4, "usage_percent": 50},
                "ram": {"total_gb": 8, "available_gb": 4, "usage_percent": 50},
                "gpu": {"available": False, "info": None, "total_memory_gb": 0, "available_memory_gb": 0},
                "resource_level": "medium",
                "recommendation": "Using default medium resource settings"
            }
    
    @staticmethod
    def _classify_resource_level(ram_available: float, gpu_memory_available: float, has_gpu: bool) -> str:
        """Classify system resource level"""
        if has_gpu and gpu_memory_available >= 6 and ram_available >= 8:
            return "high"
        elif has_gpu and gpu_memory_available >= 4 and ram_available >= 6:
            return "medium_high"
        elif ram_available >= 4:
            return "medium"
        elif ram_available >= 2:
            return "low"
        else:
            return "very_low"
    
    @staticmethod
    def _get_resource_recommendation(level: str, has_gpu: bool) -> str:
        """Get recommendation based on resource level"""
        recommendations = {
            "high": "High resources available - Can run largest models with best accuracy",
            "medium_high": "Good resources - Can run medium to large models effectively",
            "medium": "Moderate resources - Medium models recommended for balance",
            "low": "Limited resources - Lightweight models recommended",
            "very_low": "Very limited resources - Use smallest models only"
        }
        return recommendations.get(level, "Assessing resources...")


class ModelRecommendationEngine:
    """
    Recommends the most suitable detection model based on:
    - Scene complexity
    - Hardware resources
    - CPU/GPU availability
    """
    
    def __init__(self):
        """Initialize the recommendation engine"""
        self.hardware_detector = HardwareResourceDetector()
        self.complexity_analyzer = SceneComplexityAnalyzer()
    
    def recommend(
        self,
        image: Optional[np.ndarray] = None,
        scene_type: Optional[str] = None,
        priority: str = "auto"
    ) -> Dict[str, Any]:
        """
        Recommend the best model based on current conditions
        
        Args:
            image: Optional image for complexity analysis
            scene_type: Optional scene type hint (plain_text, simple, crowded, etc.)
            priority: Priority mode (auto, speed, accuracy, resources)
            
        Returns:
            Dictionary with recommendation details
        """
        # Get hardware resources
        hardware = self.hardware_detector.get_system_resources()
        
        # Analyze scene complexity if image provided
        complexity_info = None
        if image is not None:
            complexity_info = self.complexity_analyzer.analyze(image)
        
        # Determine scene characteristics
        scene_characteristics = self._analyze_scene_characteristics(image, scene_type, complexity_info)
        
        # Get candidate models filtered by hardware
        candidate_models = self._filter_models_by_hardware(hardware)
        
        # Score and rank models
        ranked_models = self._rank_models(
            candidate_models, 
            hardware, 
            scene_characteristics, 
            complexity_info,
            priority
        )
        
        # Select best model
        best_model_id = ranked_models[0]["model_id"] if ranked_models else "yolov8n"
        best_model_config = AVAILABLE_MODELS[best_model_id]
        
        # Generate recommendation details
        recommendation = {
            "recommended_model": {
                "id": best_model_id,
                "name": best_model_config["name"],
                "type": best_model_config["type"],
                "description": best_model_config["description"]
            },
            "selection_reason": self._generate_selection_reason(
                best_model_id, hardware, scene_characteristics, complexity_info, priority
            ),
            "expected_performance": self._calculate_expected_performance(
                best_model_id, hardware, scene_characteristics
            ),
            "scene_analysis": {
                "complexity": complexity_info["complexity_level"] if complexity_info else "unknown",
                "complexity_score": complexity_info["complexity_score"] if complexity_info else None,
                "scene_type": scene_characteristics.get("scene_type", "unknown"),
                "object_density": scene_characteristics.get("object_density", "unknown")
            },
            "hardware_status": {
                "resource_level": hardware["resource_level"],
                "gpu_available": hardware["gpu"]["available"],
                "gpu_memory_gb": hardware["gpu"]["available_memory_gb"],
                "ram_available_gb": hardware["ram"]["available_gb"],
                "cpu_usage_percent": hardware["cpu"]["usage_percent"]
            },
            "alternatives": [
                {
                    "id": m["model_id"],
                    "name": AVAILABLE_MODELS[m["model_id"]]["name"],
                    "reason": m.get("reason", ""),
                    "score": round(m["score"], 2)
                }
                for m in ranked_models[1:3]  # Top 2 alternatives
            ] if len(ranked_models) > 1 else []
        }
        
        return recommendation
    
    def _analyze_scene_characteristics(
        self, 
        image: Optional[np.ndarray], 
        scene_type: Optional[str],
        complexity_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze scene characteristics"""
        characteristics = {
            "scene_type": scene_type or "unknown",
            "object_density": "medium",
            "is_crowded": False,
            "is_simple": False,
            "is_plain_text": False
        }
        
        # Determine from scene type hint
        if scene_type:
            if scene_type in ["plain_text", "document", "text"]:
                characteristics["is_plain_text"] = True
                characteristics["is_simple"] = True
                characteristics["object_density"] = "very_low"
            elif scene_type in ["simple", "single_object"]:
                characteristics["is_simple"] = True
                characteristics["object_density"] = "low"
            elif scene_type in ["crowded", "many_objects", "dense"]:
                characteristics["is_crowded"] = True
                characteristics["object_density"] = "high"
        
        # Determine from complexity analysis
        if complexity_info:
            level = complexity_info["complexity_level"]
            score = complexity_info["complexity_score"]
            
            if level in ["very_low", "low"]:
                characteristics["is_simple"] = True
                characteristics["object_density"] = "low"
            elif level in ["high", "very_high"]:
                if score > 70:
                    characteristics["is_crowded"] = True
                    characteristics["object_density"] = "high"
                else:
                    characteristics["object_density"] = "medium_high"
            
            # Check for plain text indicators (low edge density but structured)
            if complexity_info["edge_density"] < 0.05 and complexity_info["brightness_std"] > 50:
                characteristics["is_plain_text"] = True
                characteristics["is_simple"] = True
        
        return characteristics
    
    def _filter_models_by_hardware(self, hardware: Dict[str, Any]) -> List[str]:
        """Filter models that can run on current hardware"""
        candidates = []
        ram_available = hardware["ram"]["available_gb"]
        gpu_available = hardware["gpu"]["available"]
        gpu_memory = hardware["gpu"]["available_memory_gb"]
        
        for model_id, config in AVAILABLE_MODELS.items():
            if not config.get("runnable", True):
                continue
            min_ram = config.get("min_ram_gb", 2)
            min_gpu_mem = config.get("min_gpu_memory_gb", 0)
            
            # Check RAM requirement
            if ram_available < min_ram:
                continue
            
            # Check GPU requirement
            if min_gpu_mem > 0 and not gpu_available:
                continue
            
            if min_gpu_mem > 0 and gpu_memory < min_gpu_mem:
                continue
            
            candidates.append(model_id)
        
        # If no candidates, fall back to most lightweight model
        if not candidates:
            candidates = ["yolov8n"]
        
        return candidates
    
    def _rank_models(
        self,
        candidates: List[str],
        hardware: Dict[str, Any],
        scene_chars: Dict[str, Any],
        complexity_info: Optional[Dict[str, Any]],
        priority: str
    ) -> List[Dict[str, Any]]:
        """Rank candidate models by suitability"""
        ranked = []
        
        for model_id in candidates:
            config = AVAILABLE_MODELS[model_id]
            score = 0
            reasons = []
            
            # Factor 1: Priority matching (35% weight)
            if priority == "speed" or priority == "realtime":
                if config["type"] == "realtime" and config.get("expected_fps_cpu", 0) >= 20:
                    score += 35
                    reasons.append("Optimized for speed")
            elif priority == "accuracy":
                if config.get("accuracy_score", 0) >= 0.75:
                    score += 35
                    reasons.append("High accuracy model")
            elif priority == "resources":
                if config.get("min_ram_gb", 4) <= 4 and config.get("min_gpu_memory_gb", 0) == 0:
                    score += 35
                    reasons.append("Low resource requirements")
            else:  # auto
                # Match scene characteristics
                if scene_chars["is_plain_text"] or scene_chars["is_simple"]:
                    if config["type"] == "realtime" and config.get("expected_fps_cpu", 0) >= 30:
                        score += 35
                        reasons.append("Fast model for simple scene")
                elif scene_chars["is_crowded"]:
                    if "crowded-scenes" in config.get("recommended_for", []) or "rt-detr" in model_id:
                        score += 35
                        reasons.append("Transformer model for crowded scenes")
                    elif config.get("accuracy_score", 0) >= 0.78:
                        score += 25
                        reasons.append("High accuracy for crowded scenes")
                else:
                    if "balanced" in config.get("recommended_for", []):
                        score += 30
                        reasons.append("Balanced performance")
            
            # Factor 2: Hardware efficiency (25% weight)
            resource_level = hardware["resource_level"]
            if resource_level in ["high", "medium_high"]:
                if config.get("min_gpu_memory_gb", 0) > 0 and hardware["gpu"]["available"]:
                    score += 25
                    reasons.append("GPU-accelerated")
            elif resource_level in ["low", "very_low"]:
                if config.get("min_ram_gb", 4) <= 2:
                    score += 25
                    reasons.append("Optimized for low resources")
            else:
                if config.get("min_ram_gb", 4) <= 4 and config.get("min_gpu_memory_gb", 0) <= 2:
                    score += 20
                    reasons.append("Moderate resource usage")
            
            # Factor 3: Scene complexity matching (25% weight)
            if complexity_info:
                complexity_level = complexity_info["complexity_level"]
                accuracy = config.get("accuracy_score", 0.5)
                
                if complexity_level in ["very_low", "low"]:
                    score += (1 - accuracy) * 25  # Prefer faster models
                    reasons.append("Appropriate for low complexity")
                elif complexity_level in ["high", "very_high"]:
                    score += accuracy * 25  # Prefer accurate models
                    reasons.append("High accuracy for complex scene")
                else:
                    score += 15
                    reasons.append("Suitable for medium complexity")
            else:
                score += 12
            
            # Factor 4: Expected performance (15% weight)
            gpu_available = hardware["gpu"]["available"]
            expected_fps = config.get("expected_fps_gpu", 0) if gpu_available else config.get("expected_fps_cpu", 0)
            
            if expected_fps >= 30:
                score += 15
                reasons.append(f"Expected {expected_fps} FPS")
            elif expected_fps >= 15:
                score += 10
                reasons.append(f"Expected {expected_fps} FPS")
            else:
                score += 5
            
            ranked.append({
                "model_id": model_id,
                "score": score,
                "reasons": reasons,
                "reason": "; ".join(reasons)
            })
        
        # Sort by score descending
        ranked.sort(key=lambda x: x["score"], reverse=True)
        
        return ranked
    
    def _generate_selection_reason(
        self,
        model_id: str,
        hardware: Dict[str, Any],
        scene_chars: Dict[str, Any],
        complexity_info: Optional[Dict[str, Any]],
        priority: str
    ) -> str:
        """Generate human-readable selection reason"""
        config = AVAILABLE_MODELS[model_id]
        reasons = []
        
        # Scene-based reasons
        if scene_chars["is_plain_text"]:
            reasons.append(f"Detected plain text/simple content")
        elif scene_chars["is_crowded"]:
            reasons.append(f"Crowded scene with many objects")
        elif scene_chars["is_simple"]:
            reasons.append(f"Simple scene with few objects")
        
        # Hardware-based reasons
        resource_level = hardware["resource_level"]
        if resource_level in ["low", "very_low"]:
            reasons.append(f"Low system resources ({hardware['ram']['available_gb']:.1f}GB RAM available)")
        elif not hardware["gpu"]["available"]:
            reasons.append("CPU-only mode")
        elif hardware["gpu"]["available"]:
            reasons.append(f"GPU available ({hardware['gpu']['info']['name'] if hardware['gpu']['info'] else 'GPU'})")
        
        # Model-specific reasons
        if "rt-detr" in model_id:
            reasons.append("Transformer architecture excels at crowded scenes")
        elif "mobilenet" in model_id:
            reasons.append("Ultra-lightweight for resource-constrained devices")
        elif config.get("accuracy_score", 0) >= 0.8:
            reasons.append(f"High accuracy model ({config['accuracy_score']*100:.0f}%)")
        elif config.get("expected_fps_cpu", 0) >= 30:
            reasons.append(f"Fast inference ({config['expected_fps_cpu']} FPS on CPU)")
        
        # Priority-based reasons
        if priority == "speed":
            reasons.append("Speed priority selected")
        elif priority == "accuracy":
            reasons.append("Accuracy priority selected")
        elif priority == "resources":
            reasons.append("Resource efficiency priority selected")
        
        return "; ".join(reasons) if reasons else "Best match for current conditions"
    
    def _calculate_expected_performance(
        self,
        model_id: str,
        hardware: Dict[str, Any],
        scene_chars: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate expected performance metrics"""
        config = AVAILABLE_MODELS[model_id]
        gpu_available = hardware["gpu"]["available"]
        
        # Base FPS
        base_fps = config.get("expected_fps_gpu", 0) if gpu_available else config.get("expected_fps_cpu", 0)
        
        # Adjust for scene complexity
        if scene_chars["is_crowded"]:
            fps_modifier = 0.7  # Crowded scenes are slower
        elif scene_chars["is_simple"]:
            fps_modifier = 1.2  # Simple scenes are faster
        else:
            fps_modifier = 1.0
        
        expected_fps = int(base_fps * fps_modifier)
        
        # Accuracy estimate
        base_accuracy = config.get("accuracy_score", 0.5)
        if scene_chars["is_crowded"] and "rt-detr" in model_id:
            accuracy_bonus = 0.1  # RT-DETR excels in crowded scenes
        elif scene_chars["is_crowded"]:
            accuracy_bonus = -0.05  # Other models struggle in crowded scenes
        else:
            accuracy_bonus = 0
        
        expected_accuracy = min(max(base_accuracy + accuracy_bonus, 0.3), 0.95)
        
        # Inference time
        expected_inference_ms = round(1000 / expected_fps, 1) if expected_fps > 0 else "N/A"
        
        return {
            "expected_fps": expected_fps,
            "expected_accuracy": round(expected_accuracy * 100, 1),
            "expected_inference_time_ms": expected_inference_ms,
            "device": "GPU" if gpu_available else "CPU",
            "memory_usage_mb": round(config.get("min_ram_gb", 2) * 512, 0),  # Rough estimate
            "recommended_confidence_threshold": 0.45 if scene_chars["is_crowded"] else 0.5
        }


# Singleton instances
_adaptive_switcher: Optional[AdaptiveModelSwitcher] = None
_recommendation_engine: Optional[ModelRecommendationEngine] = None


def get_adaptive_switcher() -> AdaptiveModelSwitcher:
    """Get or create adaptive model switcher singleton"""
    global _adaptive_switcher
    if _adaptive_switcher is None:
        _adaptive_switcher = AdaptiveModelSwitcher()
    return _adaptive_switcher


def get_recommendation_engine() -> ModelRecommendationEngine:
    """Get or create recommendation engine singleton"""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = ModelRecommendationEngine()
    return _recommendation_engine
