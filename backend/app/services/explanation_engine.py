"""
AI Decision Explanation Engine
Provides human-readable explanations for AI model decisions,
detection confidence, model selection, and scene analysis.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExplanationComponent:
    """Represents a component of an explanation"""
    component_type: str  # 'factor', 'evidence', 'reasoning', 'alternative'
    title: str
    description: str
    confidence_impact: float  # -1.0 to 1.0
    metadata: Dict[str, Any]


class DetectionExplainer:
    """Explains individual detection decisions"""
    
    def __init__(self):
        """Initialize detection explainer"""
        self.confidence_factors = {
            "high_confidence": {
                "threshold": 0.8,
                "explanation": "Very high confidence detection with clear visual features"
            },
            "medium_confidence": {
                "threshold": 0.6,
                "explanation": "Good confidence detection with recognizable features"
            },
            "low_confidence": {
                "threshold": 0.4,
                "explanation": "Moderate confidence detection, may require verification"
            },
            "very_low_confidence": {
                "threshold": 0.0,
                "explanation": "Low confidence detection, likely false positive"
            }
        }
        
        # Size-based explanations
        self.size_factors = {
            "large": {"min_area": 10000, "explanation": "Large object with substantial visual presence"},
            "medium": {"min_area": 1000, "explanation": "Medium-sized object with clear boundaries"},
            "small": {"min_area": 100, "explanation": "Small object, may have limited features"},
            "tiny": {"min_area": 0, "explanation": "Very small object, detection challenging"}
        }
        
        # Position-based explanations
        self.position_factors = {
            "center": {"explanation": "Object is in the center of the frame, optimal detection position"},
            "edge": {"explanation": "Object is near the edge, may be partially occluded"},
            "corner": {"explanation": "Object is in corner, limited context available"}
        }
    
    def explain_detection(
        self, 
        detection: Dict[str, Any], 
        image_dimensions: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single detection
        
        Args:
            detection: Detection result dictionary
            image_dimensions: Image width and height
            
        Returns:
            Explanation dictionary
        """
        confidence = detection["confidence"]
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        gradcam_activation = (detection.get("metadata") or {}).get("gradcam_activation")
        
        # Calculate object size
        area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
        image_area = image_dimensions["width"] * image_dimensions["height"]
        relative_size = area / image_area
        
        # Determine size category
        size_category = self._get_size_category(area)
        
        # Determine position
        position = self._get_position(bbox, image_dimensions)
        
        # Build explanation components
        components = []
        
        # Confidence explanation
        confidence_explanation = self._get_confidence_explanation(confidence)
        components.append({
            "type": "confidence",
            "title": "Detection Confidence",
            "description": confidence_explanation["explanation"],
            "value": confidence,
            "category": confidence_explanation["category"]
        })
        
        # Size explanation
        components.append({
            "type": "size",
            "title": "Object Size",
            "description": size_category["explanation"],
            "value": {
                "area": area,
                "relative_size": round(relative_size * 100, 2),
                "category": size_category["category"]
            }
        })
        
        # Position explanation
        components.append({
            "type": "position",
            "title": "Object Position",
            "description": position["explanation"],
            "value": position["category"]
        })
        
        # Class-specific explanation
        class_explanation = (
            self._get_gradcam_class_explanation(class_name, confidence, gradcam_activation)
            if gradcam_activation
            else self._get_class_explanation(class_name, confidence)
        )
        components.append({
            "type": "class_specific",
            "title": f"Why '{class_name}'?",
            "description": class_explanation,
            "value": class_name
        })

        if gradcam_activation:
            components.append({
                "type": "gradcam_activation",
                "title": "GradCAM Focus",
                "description": self._get_gradcam_focus_description(class_name, confidence, gradcam_activation),
                "value": gradcam_activation
            })
        
        # Generate overall explanation
        overall_explanation = (
            self._generate_gradcam_overall_explanation(class_name, confidence, gradcam_activation)
            if gradcam_activation
            else self._generate_overall_explanation(class_name, confidence, size_category, position, components)
        )
        
        return {
            "detection_id": f"{class_name}_{bbox['x1']}_{bbox['y1']}",
            "class_name": class_name,
            "confidence": confidence,
            "overall_explanation": overall_explanation,
            "components": components,
            "factors": {
                "positive": [c for c in components if self._is_positive_factor(c)],
                "negative": [c for c in components if not self._is_positive_factor(c)]
            },
            "recommendation": self._get_recommendation(confidence, size_category, position)
        }

    def _get_gradcam_class_explanation(
        self,
        class_name: str,
        confidence: float,
        activation: Dict[str, Any]
    ) -> str:
        region = activation.get("primary_region", "the highlighted region")
        mean_activation = round(float(activation.get("mean_activation", 0)) * 100, 1)
        return (
            f"Model focused on {region} region ({mean_activation}% activation) "
            f"to detect '{class_name}' with {confidence:.2f} confidence"
        )

    def _get_gradcam_focus_description(
        self,
        class_name: str,
        confidence: float,
        activation: Dict[str, Any]
    ) -> str:
        region = activation.get("primary_region", "unknown")
        max_activation = round(float(activation.get("max_activation", 0)) * 100, 1)
        return (
            f"The strongest GradCAM response for '{class_name}' appears in the "
            f"{region} region, peaking at {max_activation}% activation."
        )

    def _generate_gradcam_overall_explanation(
        self,
        class_name: str,
        confidence: float,
        activation: Dict[str, Any]
    ) -> str:
        region = activation.get("primary_region", "the highlighted region")
        mean_activation = round(float(activation.get("mean_activation", 0)) * 100, 1)
        return (
            f"Model focused on {region} region ({mean_activation}% activation) "
            f"to detect '{class_name}' with {confidence:.2f} confidence."
        )
    
    def _get_confidence_explanation(self, confidence: float) -> Dict[str, Any]:
        """Get explanation for confidence level"""
        for level, info in self.confidence_factors.items():
            if confidence >= info["threshold"]:
                return {
                    "category": level,
                    "explanation": info["explanation"],
                    "confidence": confidence
                }
        return {
            "category": "very_low_confidence",
            "explanation": "Very low confidence, likely incorrect detection"
        }
    
    def _get_size_category(self, area: float) -> Dict[str, Any]:
        """Get size category explanation"""
        for category, info in self.size_factors.items():
            if area >= info["min_area"]:
                return {
                    "category": category,
                    "explanation": info["explanation"],
                    "area": area
                }
        return {
            "category": "tiny",
            "explanation": "Very small object, detection challenging"
        }
    
    def _get_position(self, bbox: Dict[str, float], dims: Dict[str, int]) -> Dict[str, Any]:
        """Get position category"""
        center_x = (bbox["x1"] + bbox["x2"]) / 2
        center_y = (bbox["y1"] + bbox["y2"]) / 2
        
        width_ratio = center_x / dims["width"]
        height_ratio = center_y / dims["height"]
        
        # Check if near edges
        near_edge = (
            width_ratio < 0.15 or width_ratio > 0.85 or
            height_ratio < 0.15 or height_ratio > 0.85
        )
        
        # Check if in corner
        in_corner = (
            (width_ratio < 0.2 and height_ratio < 0.2) or
            (width_ratio > 0.8 and height_ratio < 0.2) or
            (width_ratio < 0.2 and height_ratio > 0.8) or
            (width_ratio > 0.8 and height_ratio > 0.8)
        )
        
        if in_corner:
            return {"category": "corner", "explanation": self.position_factors["corner"]["explanation"]}
        elif near_edge:
            return {"category": "edge", "explanation": self.position_factors["edge"]["explanation"]}
        else:
            return {"category": "center", "explanation": self.position_factors["center"]["explanation"]}
    
    def _get_class_explanation(self, class_name: str, confidence: float) -> str:
        """Get class-specific explanation"""
        # Common visual features for different classes
        class_features = {
            "person": "upright posture, human-like proportions, recognizable body parts",
            "car": "rectangular shape, wheels, windows, typical vehicle proportions",
            "dog": "four-legged stance, fur texture, animal proportions",
            "cat": "smaller feline features, pointed ears, tail",
            "chair": "seat structure, backrest, legs in typical arrangement",
            "table": "flat surface, supporting legs, rectangular or round shape",
            "laptop": "rectangular screen, keyboard area, clamshell design",
            "cell phone": "small rectangular shape, screen reflection, handheld size",
            "cup": "cylindrical shape, open top, handle (if visible)",
            "book": "rectangular shape, flat surfaces, page edges",
            "tv": "large rectangular screen, thin profile, stand or wall mount"
        }
        
        base_explanation = class_features.get(class_name, "recognizable visual patterns and features")
        
        if confidence > 0.8:
            return f"Strong match for '{class_name}' based on {base_explanation}"
        elif confidence > 0.6:
            return f"Moderate match for '{class_name}' with {base_explanation}"
        elif confidence > 0.4:
            return f"Weak match for '{class_name}', some features align with {base_explanation}"
        else:
            return f"Uncertain detection of '{class_name}', limited feature match with {base_explanation}"
    
    def _generate_feature_importance(self, detection: Dict[str, Any], image_dimensions: Dict[str, int]) -> Dict[str, Any]:
        """Generate feature importance scores for the detection"""
        class_name = detection["class_name"]
        bbox = detection["bbox"]
        confidence = detection["confidence"]
        
        # Define important regions based on class
        class_regions = {
            "person": [
                {"region": "Face", "importance": 0.35, "description": "Facial features are strong indicators"},
                {"region": "Body", "importance": 0.30, "description": "Body shape and posture"},
                {"region": "Limbs", "importance": 0.20, "description": "Arms and legs positioning"},
                {"region": "Background", "importance": 0.15, "description": "Context clues from surroundings"}
            ],
            "car": [
                {"region": "Wheels", "importance": 0.30, "description": "Circular wheel shapes"},
                {"region": "Body", "importance": 0.35, "description": "Vehicle body shape"},
                {"region": "Windows", "importance": 0.20, "description": "Window outlines"},
                {"region": "Background", "importance": 0.15, "description": "Road/parking context"}
            ],
            "dog": [
                {"region": "Head", "importance": 0.30, "description": "Facial features, ears"},
                {"region": "Body", "importance": 0.35, "description": "Body shape, fur texture"},
                {"region": "Legs", "importance": 0.20, "description": "Four-legged stance"},
                {"region": "Tail", "importance": 0.15, "description": "Tail presence and shape"}
            ],
            "cat": [
                {"region": "Head", "importance": 0.35, "description": "Pointed ears, facial features"},
                {"region": "Body", "importance": 0.30, "description": "Feline body shape"},
                {"region": "Tail", "importance": 0.20, "description": "Long tail"},
                {"region": "Paws", "importance": 0.15, "description": "Small paws"}
            ],
            "chair": [
                {"region": "Seat", "importance": 0.35, "description": "Flat seating surface"},
                {"region": "Backrest", "importance": 0.30, "description": "Vertical support"},
                {"region": "Legs", "importance": 0.25, "description": "Support legs"},
                {"region": "Background", "importance": 0.10, "description": "Furniture context"}
            ],
            "table": [
                {"region": "Surface", "importance": 0.40, "description": "Flat top surface"},
                {"region": "Legs", "importance": 0.30, "description": "Support structure"},
                {"region": "Edges", "importance": 0.20, "description": "Table edges"},
                {"region": "Background", "importance": 0.10, "description": "Room context"}
            ],
            "laptop": [
                {"region": "Screen", "importance": 0.40, "description": "Display area"},
                {"region": "Keyboard", "importance": 0.35, "description": "Key layout"},
                {"region": "Body", "importance": 0.15, "description": "Clamshell body"},
                {"region": "Background", "importance": 0.10, "description": "Desk/workspace context"}
            ],
            "cell phone": [
                {"region": "Screen", "importance": 0.45, "description": "Display area"},
                {"region": "Body", "importance": 0.30, "description": "Rectangular body"},
                {"region": "Edges", "importance": 0.15, "description": "Phone edges"},
                {"region": "Background", "importance": 0.10, "description": "Hand/context"}
            ]
        }
        
        # Default regions for unknown classes
        default_regions = [
            {"region": "Main Object", "importance": 0.50, "description": "Primary object features"},
            {"region": "Edges", "importance": 0.25, "description": "Object boundaries"},
            {"region": "Texture", "importance": 0.15, "description": "Surface patterns"},
            {"region": "Background", "importance": 0.10, "description": "Context clues"}
        ]
        
        regions = class_regions.get(class_name, default_regions)
        
        # Adjust importance based on confidence
        adjusted_regions = []
        for region in regions:
            adjusted_regions.append({
                "region": region["region"],
                "importance": round(region["importance"] * (0.8 + confidence * 0.4), 2),
                "description": region["description"],
                "heatmap_intensity": round(region["importance"] * confidence, 3)
            })
        
        # Sort by importance
        adjusted_regions.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "features": adjusted_regions,
            "primary_feature": adjusted_regions[0]["region"] if adjusted_regions else "Unknown",
            "confidence_contribution": round(confidence * 0.7, 3),
            "context_contribution": round((1 - confidence) * 0.3, 3)
        }
    
    def _identify_important_regions(self, detection: Dict[str, Any], image_dimensions: Dict[str, int]) -> Dict[str, Any]:
        """Identify the most important regions for the detection"""
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        
        # Calculate relative positions within the bounding box
        width = bbox["x2"] - bbox["x1"]
        height = bbox["y2"] - bbox["y1"]
        
        # Define regions as percentages of the bounding box
        regions = {
            "Top": {"x": 0.5, "y": 0.2, "w": 0.6, "h": 0.25},
            "Center": {"x": 0.5, "y": 0.5, "w": 0.7, "h": 0.4},
            "Bottom": {"x": 0.5, "y": 0.8, "w": 0.6, "h": 0.25},
            "Left": {"x": 0.25, "y": 0.5, "w": 0.25, "h": 0.6},
            "Right": {"x": 0.75, "y": 0.5, "w": 0.25, "h": 0.6}
        }
        
        # Class-specific important regions
        class_important = {
            "person": ["Top", "Center"],  # Face and body
            "car": ["Center", "Bottom"],  # Body and wheels
            "dog": ["Center", "Top"],  # Body and head
            "cat": ["Center", "Top"],  # Body and head
            "chair": ["Center", "Bottom"],  # Seat and legs
            "table": ["Center", "Bottom"],  # Surface and legs
            "laptop": ["Top", "Center"],  # Screen and keyboard
            "cell phone": ["Center", "Top"],  # Screen
            "cup": ["Center", "Top"],  # Body and opening
            "book": ["Center"],  # Cover/pages
            "tv": ["Center", "Top"]  # Screen
        }
        
        important = class_important.get(class_name, ["Center"])
        
        # Generate pixel coordinates for important regions
        important_regions = []
        for region_name in important:
            region = regions[region_name]
            important_regions.append({
                "name": region_name,
                "bbox": {
                    "x1": bbox["x1"] + (region["x"] - region["w"]/2) * width,
                    "y1": bbox["y1"] + (region["y"] - region["h"]/2) * height,
                    "x2": bbox["x1"] + (region["x"] + region["w"]/2) * width,
                    "y2": bbox["y1"] + (region["y"] + region["h"]/2) * height
                },
                "relative_position": {
                    "x": region["x"],
                    "y": region["y"],
                    "width": region["w"],
                    "height": region["h"]
                }
            })
        
        return {
            "important_regions": important_regions,
            "primary_region": important_regions[0]["name"] if important_regions else "Center",
            "region_count": len(important_regions)
        }
    
    def _generate_overall_explanation(
        self,
        class_name: str,
        confidence: float,
        size: Dict[str, Any],
        position: Dict[str, Any],
        components: List[Dict[str, Any]]
    ) -> str:
        """Generate overall explanation"""
        parts = [f"The AI detected a '{class_name}'"]
        
        # Add confidence context
        if confidence > 0.8:
            parts.append("with high confidence")
        elif confidence > 0.6:
            parts.append("with moderate confidence")
        elif confidence > 0.4:
            parts.append("with low confidence")
        else:
            parts.append("with very low confidence")
        
        # Add size context
        if size["category"] in ["large", "medium"]:
            parts.append(f", which is {size['category']} in size")
        else:
            parts.append(f", which is {size['category']} and may be harder to detect")
        
        # Add position context
        if position["category"] == "center":
            parts.append(", positioned optimally in the frame")
        elif position["category"] == "edge":
            parts.append(", located near the edge of the frame")
        else:
            parts.append(", located in a corner with limited context")
        
        return "".join(parts) + "."
    
    def _is_positive_factor(self, component: Dict[str, Any]) -> bool:
        """Determine if a factor is positive for detection quality"""
        if component["type"] == "confidence":
            return component["value"] > 0.6
        elif component["type"] == "size":
            return component["value"]["category"] in ["large", "medium"]
        elif component["type"] == "position":
            return component["value"] == "center"
        return True
    
    def _get_recommendation(
        self, 
        confidence: float, 
        size: Dict[str, Any], 
        position: Dict[str, Any]
    ) -> str:
        """Get recommendation based on detection quality"""
        if confidence > 0.8:
            return "Detection is reliable, no action needed"
        elif confidence > 0.6:
            if size["category"] in ["small", "tiny"]:
                return "Consider using a higher resolution model for small objects"
            elif position["category"] in ["edge", "corner"]:
                return "Object position may affect accuracy, consider reframing"
            else:
                return "Detection is reasonably reliable"
        elif confidence > 0.4:
            return "Consider manual verification or use a more accurate model"
        else:
            return "High probability of false positive, recommend verification"


class ModelSelectionExplainer:
    """Explains why a particular model was selected"""
    
    def __init__(self):
        """Initialize model selection explainer"""
        self.model_characteristics = {
            "yolov8n": {
                "name": "YOLOv8 Nano",
                "strengths": ["fastest inference", "smallest memory footprint", "real-time performance"],
                "weaknesses": ["lower accuracy on small objects", "may miss fine details"],
                "best_for": ["real-time applications", "low-power devices", "simple scenes"]
            },
            "yolov8s": {
                "name": "YOLOv8 Small",
                "strengths": ["good speed-accuracy balance", "efficient resource usage"],
                "weaknesses": ["may struggle with very small objects"],
                "best_for": ["balanced workloads", "medium complexity scenes"]
            },
            "yolov8m": {
                "name": "YOLOv8 Medium",
                "strengths": ["better accuracy than small models", "handles complex scenes well"],
                "weaknesses": ["slower inference", "higher memory usage"],
                "best_for": ["accuracy-focused applications", "complex scenes"]
            },
            "yolov8l": {
                "name": "YOLOv8 Large",
                "strengths": ["high accuracy", "excellent for detailed analysis"],
                "weaknesses": ["slow inference", "high computational requirements"],
                "best_for": ["high-accuracy requirements", "detailed scene analysis"]
            },
            "yolov8x": {
                "name": "YOLOv8 Extra Large",
                "strengths": ["highest accuracy", "best for challenging detections"],
                "weaknesses": ["slowest inference", "highest resource usage"],
                "best_for": ["maximum accuracy requirements", "research applications"]
            }
        }
    
    def explain_model_selection(
        self, 
        model_id: str, 
        selection_reasons: List[str],
        scene_complexity: Optional[Dict[str, Any]] = None,
        use_case: str = "auto"
    ) -> Dict[str, Any]:
        """
        Explain why a model was selected
        
        Args:
            model_id: Selected model identifier
            selection_reasons: List of reasons for selection
            scene_complexity: Scene complexity information
            use_case: Use case type
            
        Returns:
            Explanation dictionary
        """
        model_info = self.model_characteristics.get(model_id, {
            "name": model_id,
            "strengths": [],
            "weaknesses": [],
            "best_for": []
        })
        
        # Build explanation
        explanation_parts = [f"Selected {model_info['name']}"]
        
        # Add use case context
        if use_case == "realtime":
            explanation_parts.append("optimized for real-time performance")
        elif use_case == "accuracy":
            explanation_parts.append("optimized for maximum accuracy")
        elif use_case == "balanced":
            explanation_parts.append("balanced for speed and accuracy")
        
        # Add scene complexity context
        if scene_complexity:
            complexity = scene_complexity.get("complexity_level", "unknown")
            if complexity in ["high", "very_high"]:
                explanation_parts.append("to handle complex scene details")
            elif complexity in ["low", "very_low"]:
                explanation_parts.append("sufficient for simple scene")
        
        primary_reason = ". ".join(explanation_parts) + "."
        
        # Detailed reasons
        detailed_reasons = []
        for reason in selection_reasons[:5]:
            detailed_reasons.append(reason)
        
        # Trade-offs
        trade_offs = []
        if "fast" in model_id or "nano" in model_id.lower() or "n" in model_id:
            trade_offs.append("Prioritizing speed over maximum accuracy")
        elif "large" in model_id or "x" in model_id:
            trade_offs.append("Prioritizing accuracy over inference speed")
        else:
            trade_offs.append("Balancing speed and accuracy")
        
        return {
            "model_id": model_id,
            "model_name": model_info["name"],
            "primary_explanation": primary_reason,
            "detailed_reasons": detailed_reasons,
            "trade_offs": trade_offs,
            "model_strengths": model_info["strengths"],
            "model_weaknesses": model_info["weaknesses"],
            "recommendations": model_info["best_for"],
            "use_case": use_case,
            "scene_complexity": scene_complexity
        }


class SceneAnalysisExplainer:
    """Explains overall scene analysis and patterns"""
    
    def explain_scene_analysis(
        self, 
        detections: List[Dict[str, Any]], 
        image_dimensions: Dict[str, int],
        model_used: str,
        adaptive_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Provide comprehensive scene analysis explanation
        
        Args:
            detections: List of detections
            image_dimensions: Image dimensions
            model_used: Model used for detection
            adaptive_analysis: Analysis from adaptive AI system
            
        Returns:
            Scene analysis explanation
        """
        num_objects = len(detections)
        unique_classes = list(set(d["class_name"] for d in detections))
        
        # Scene complexity assessment
        complexity = self._assess_scene_complexity(detections, image_dimensions)
        
        # Dominant themes
        themes = self._identify_scene_themes(detections)
        
        # Spatial analysis
        spatial_analysis = self._analyze_spatial_distribution(detections, image_dimensions)
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence_distribution(detections)
        
        # Generate narrative explanation
        narrative = self._generate_scene_narrative(
            num_objects, unique_classes, complexity, themes, confidence_analysis
        )
        
        return {
            "summary": {
                "object_count": num_objects,
                "unique_classes": len(unique_classes),
                "classes_detected": unique_classes,
                "complexity_level": complexity["level"],
                "dominant_themes": themes
            },
            "complexity_analysis": complexity,
            "spatial_analysis": spatial_analysis,
            "confidence_analysis": confidence_analysis,
            "narrative_explanation": narrative,
            "model_assessment": self._assess_model_performance(detections, model_used),
            "recommendations": self._generate_scene_recommendations(
                complexity, detections, model_used, adaptive_analysis
            )
        }
    
    def _assess_scene_complexity(
        self, 
        detections: List[Dict[str, Any]], 
        image_dimensions: Dict[str, int]
    ) -> Dict[str, Any]:
        """Assess scene complexity"""
        num_objects = len(detections)
        unique_classes = len(set(d["class_name"] for d in detections))
        
        # Calculate average confidence
        avg_confidence = sum(d["confidence"] for d in detections) / max(num_objects, 1)
        
        # Determine complexity level
        if num_objects <= 3 and unique_classes <= 2:
            level = "low"
            description = "Simple scene with few objects"
        elif num_objects <= 8 and unique_classes <= 5:
            level = "medium"
            description = "Moderately complex scene"
        else:
            level = "high"
            description = "Complex scene with many objects and variety"
        
        return {
            "level": level,
            "description": description,
            "factors": {
                "object_count": num_objects,
                "class_diversity": unique_classes,
                "average_confidence": round(avg_confidence, 3)
            }
        }
    
    def _identify_scene_themes(self, detections: List[Dict[str, Any]]) -> List[str]:
        """Identify thematic elements in the scene"""
        classes = [d["class_name"] for d in detections]
        themes = []
        
        # Check for common themes
        if any(c in classes for c in ["person", "chair", "couch", "tv"]):
            themes.append("indoor/living space")
        if any(c in classes for c in ["car", "truck", "traffic light", "stop sign"]):
            themes.append("outdoor/traffic")
        if any(c in classes for c in ["laptop", "keyboard", "mouse", "book"]):
            themes.append("workspace/office")
        if any(c in classes for c in ["dining table", "chair", "cup", "bowl", "fork", "knife"]):
            themes.append("dining/food")
        if any(c in classes for c in ["bed", "teddy bear", "book"]):
            themes.append("bedroom")
        if any(c in classes for c in ["dog", "cat", "horse", "bird"]):
            themes.append("animals present")
        if any(c in classes for c in ["refrigerator", "microwave", "oven", "sink"]):
            themes.append("kitchen")
        
        return themes if themes else ["mixed/general"]
    
    def _analyze_spatial_distribution(
        self, 
        detections: List[Dict[str, Any]], 
        image_dimensions: Dict[str, int]
    ) -> Dict[str, Any]:
        """Analyze spatial distribution of objects"""
        if not detections:
            return {"distribution": "empty", "description": "No objects detected"}
        
        # Calculate centers
        centers = []
        for det in detections:
            bbox = det["bbox"]
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2
            centers.append((center_x, center_y))
        
        # Analyze distribution
        x_coords = [c[0] for c in centers]
        y_coords = [c[1] for c in centers]
        
        x_range = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 0
        y_range = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 0
        
        if x_range < image_dimensions["width"] * 0.3 and y_range < image_dimensions["height"] * 0.3:
            distribution = "clustered"
            description = "Objects are clustered in a specific area"
        elif x_range > image_dimensions["width"] * 0.7 or y_range > image_dimensions["height"] * 0.7:
            distribution = "dispersed"
            description = "Objects are spread across the scene"
        else:
            distribution = "distributed"
            description = "Objects are moderately distributed"
        
        return {
            "distribution": distribution,
            "description": description,
            "spread_x": round(x_range / image_dimensions["width"] * 100, 1),
            "spread_y": round(y_range / image_dimensions["height"] * 100, 1)
        }
    
    def _analyze_confidence_distribution(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence distribution"""
        if not detections:
            return {"average": 0, "distribution": "none"}
        
        confidences = [d["confidence"] for d in detections]
        avg_conf = sum(confidences) / len(confidences)
        
        high_conf = sum(1 for c in confidences if c > 0.8)
        med_conf = sum(1 for c in confidences if 0.5 < c <= 0.8)
        low_conf = sum(1 for c in confidences if c <= 0.5)
        
        if high_conf > len(confidences) * 0.7:
            distribution = "high"
            description = "Most detections have high confidence"
        elif low_conf > len(confidences) * 0.3:
            distribution = "mixed"
            description = "Mix of high and low confidence detections"
        else:
            distribution = "moderate"
            description = "Most detections have moderate confidence"
        
        return {
            "average": round(avg_conf, 3),
            "distribution": distribution,
            "description": description,
            "breakdown": {
                "high_confidence": high_conf,
                "medium_confidence": med_conf,
                "low_confidence": low_conf
            }
        }
    
    def _generate_scene_narrative(
        self,
        num_objects: int,
        unique_classes: List[str],
        complexity: Dict[str, Any],
        themes: List[str],
        confidence_analysis: Dict[str, Any]
    ) -> str:
        """Generate narrative explanation of the scene"""
        parts = []
        
        # Opening
        if num_objects == 0:
            return "No objects were detected in this scene."
        
        parts.append(f"The scene contains {num_objects} detected objects")
        
        # Class diversity
        if len(unique_classes) == 1:
            parts.append(f", all of which are '{unique_classes[0]}'")
        else:
            parts.append(f" spanning {len(unique_classes)} different classes")
        
        # Complexity
        parts.append(f". This is a {complexity['level']} complexity scene")
        
        # Themes
        if themes and themes != ["mixed/general"]:
            parts.append(f", suggesting a {', '.join(themes)} environment")
        
        # Confidence
        parts.append(f". Detection confidence is generally {confidence_analysis['distribution']}")
        
        return "".join(parts) + "."
    
    def _assess_model_performance(
        self, 
        detections: List[Dict[str, Any]], 
        model_used: str
    ) -> Dict[str, Any]:
        """Assess how well the model performed"""
        if not detections:
            return {
                "assessment": "No detections to assess",
                "suggestions": ["Try a more sensitive model", "Check image quality"]
            }
        
        avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
        
        if avg_confidence > 0.8:
            assessment = "excellent"
            description = "Model is performing very well with high confidence detections"
        elif avg_confidence > 0.6:
            assessment = "good"
            description = "Model is performing adequately"
        elif avg_confidence > 0.4:
            assessment = "moderate"
            description = "Model performance is moderate, some detections uncertain"
        else:
            assessment = "poor"
            description = "Model is struggling with this scene"
        
        suggestions = []
        if assessment in ["moderate", "poor"]:
            suggestions.append("Consider using a larger, more accurate model")
            suggestions.append("Check if image quality is affecting detection")
        if len(detections) == 0:
            suggestions.append("Lower confidence threshold may help")
            suggestions.append("Try a different model architecture")
        
        return {
            "assessment": assessment,
            "description": description,
            "average_confidence": round(avg_confidence, 3),
            "suggestions": suggestions
        }
    
    def _generate_scene_recommendations(
        self,
        complexity: Dict[str, Any],
        detections: List[Dict[str, Any]],
        model_used: str,
        adaptive_analysis: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate recommendations for the scene"""
        recommendations = []
        
        # Complexity-based recommendations
        if complexity["level"] == "high":
            recommendations.append("Consider using a more accurate model for complex scenes")
            if len(detections) > 10:
                recommendations.append("High object count - ensure adequate processing time")
        elif complexity["level"] == "low":
            recommendations.append("Current model is well-suited for this simple scene")
        
        # Confidence-based recommendations
        if detections:
            avg_conf = sum(d["confidence"] for d in detections) / len(detections)
            if avg_conf < 0.6:
                recommendations.append("Consider manual verification due to lower confidence")
        
        # Adaptive AI recommendations
        if adaptive_analysis:
            if adaptive_analysis.get("should_switch"):
                recommendations.append(f"Adaptive AI suggests switching to {adaptive_analysis.get('selected_model', 'another model')}")
        
        return recommendations


class ExplanationEngine:
    """
    Main explanation engine that coordinates all explanation services
    """
    
    def __init__(self):
        """Initialize explanation engine"""
        self.detection_explainer = DetectionExplainer()
        self.model_explainer = ModelSelectionExplainer()
        self.scene_explainer = SceneAnalysisExplainer()
        
        # Explanation history
        self.explanation_history: List[Dict[str, Any]] = []
    
    def explain_detection(
        self, 
        detection: Dict[str, Any], 
        image_dimensions: Dict[str, int]
    ) -> Dict[str, Any]:
        """Explain a single detection"""
        explanation = self.detection_explainer.explain_detection(detection, image_dimensions)
        self._record_explanation("detection", explanation)
        return explanation
    
    def explain_model_selection(
        self,
        model_id: str,
        selection_reasons: List[str],
        scene_complexity: Optional[Dict[str, Any]] = None,
        use_case: str = "auto"
    ) -> Dict[str, Any]:
        """Explain model selection decision"""
        explanation = self.model_explainer.explain_model_selection(
            model_id, selection_reasons, scene_complexity, use_case
        )
        self._record_explanation("model_selection", explanation)
        return explanation
    
    def explain_scene_analysis(
        self,
        detections: List[Dict[str, Any]],
        image_dimensions: Dict[str, int],
        model_used: str,
        adaptive_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Provide comprehensive scene analysis explanation"""
        explanation = self.scene_explainer.explain_scene_analysis(
            detections, image_dimensions, model_used, adaptive_analysis
        )
        self._record_explanation("scene_analysis", explanation)
        return explanation
    
    def explain_full_detection_result(
        self,
        detection_result: Dict[str, Any],
        adaptive_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Provide comprehensive explanation for a full detection result
        
        Args:
            detection_result: Full detection result from the API
            adaptive_analysis: Analysis from adaptive AI system
            
        Returns:
            Comprehensive explanation
        """
        detections = detection_result.get("results", [])
        image_dimensions = {
            "width": detection_result.get("original_width", 0),
            "height": detection_result.get("original_height", 0)
        }
        model_used = detection_result.get("model_name", "unknown")
        
        # Scene complexity from adaptive analysis
        scene_complexity = None
        if adaptive_analysis and "scene_complexity" in adaptive_analysis:
            scene_complexity = adaptive_analysis["scene_complexity"]
        
        explanations = {
            "scene_analysis": self.explain_scene_analysis(
                detections, image_dimensions, model_used, adaptive_analysis
            ),
            "model_selection": self.explain_model_selection(
                model_used,
                adaptive_analysis.get("selection_reasons", []) if adaptive_analysis else [],
                scene_complexity,
                adaptive_analysis.get("use_case", "auto") if adaptive_analysis else "auto"
            ),
            "detections": [
                self.explain_detection(det, image_dimensions)
                for det in detections[:10]  # Limit to first 10 detections
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return explanations
    
    def _record_explanation(self, explanation_type: str, explanation: Dict[str, Any]):
        """Record explanation in history"""
        self.explanation_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": explanation_type,
            "explanation": explanation
        })
        
        # Keep only last 500 explanations
        if len(self.explanation_history) > 500:
            self.explanation_history = self.explanation_history[-500:]
    
    def get_explanation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent explanation history"""
        return self.explanation_history[-limit:]
    
    def clear(self):
        """Clear explanation history"""
        self.explanation_history.clear()


# Singleton instance
_explanation_engine: Optional[ExplanationEngine] = None


def get_explanation_engine() -> ExplanationEngine:
    """Get or create explanation engine singleton"""
    global _explanation_engine
    if _explanation_engine is None:
        _explanation_engine = ExplanationEngine()
    return _explanation_engine
