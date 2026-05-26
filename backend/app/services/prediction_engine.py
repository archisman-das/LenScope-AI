"""
Prediction Engine Service
Provides predictive analytics and forecasting based on detection history,
object trajectories, and scene patterns.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Represents a prediction result"""
    prediction_type: str
    confidence: float
    predicted_class: Optional[str]
    predicted_location: Optional[Dict[str, float]]
    time_horizon: float  # seconds into the future
    reasoning: List[str]
    metadata: Dict[str, Any]


class TrajectoryPredictor:
    """Predicts future positions of objects based on movement patterns"""
    
    def __init__(self):
        """Initialize trajectory predictor"""
        self.trajectory_history: Dict[str, List[Tuple[float, float, datetime]]] = defaultdict(list)
        
    def update_trajectory(self, track_id: str, bbox: Dict[str, float], timestamp: datetime):
        """Update trajectory history for a track"""
        center_x = (bbox["x1"] + bbox["x2"]) / 2
        center_y = (bbox["y1"] + bbox["y2"]) / 2
        self.trajectory_history[track_id].append((center_x, center_y, timestamp))
        
        # Keep only last 50 positions
        if len(self.trajectory_history[track_id]) > 50:
            self.trajectory_history[track_id] = self.trajectory_history[track_id][-50:]
    
    def predict_future_position(
        self, 
        track_id: str, 
        time_horizon: float = 1.0,
        method: str = "linear"
    ) -> Optional[Dict[str, Any]]:
        """
        Predict future position of an object
        
        Args:
            track_id: Track identifier
            time_horizon: Seconds into the future to predict
            method: Prediction method (linear, polynomial, kalman)
            
        Returns:
            Predicted position and confidence
        """
        if track_id not in self.trajectory_history:
            return None
        
        trajectory = self.trajectory_history[track_id]
        
        if len(trajectory) < 3:
            return None  # Not enough data
        
        # Extract positions and timestamps
        positions = [(x, y) for x, y, _ in trajectory]
        timestamps = [t for _, _, t in trajectory]
        
        # Calculate time deltas
        time_deltas = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]).total_seconds()
            time_deltas.append(delta)
        
        avg_time_delta = np.mean(time_deltas) if time_deltas else 0.1
        
        # Predict based on method
        if method == "linear":
            return self._linear_prediction(positions, time_deltas, time_horizon, avg_time_delta)
        elif method == "polynomial":
            return self._polynomial_prediction(positions, time_deltas, time_horizon, avg_time_delta)
        else:
            return self._linear_prediction(positions, time_deltas, time_horizon, avg_time_delta)
    
    def _linear_prediction(
        self, 
        positions: List[Tuple[float, float]], 
        time_deltas: List[float],
        time_horizon: float,
        avg_time_delta: float
    ) -> Dict[str, Any]:
        """Simple linear prediction based on recent velocity"""
        if len(positions) < 2:
            return None
        
        # Calculate recent velocity (last 5 positions)
        recent_positions = positions[-5:]
        if len(recent_positions) < 2:
            return None
        
        # Calculate average velocity
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            velocities_x.append(dx)
            velocities_y.append(dy)
        
        avg_vx = np.mean(velocities_x)
        avg_vy = np.mean(velocities_y)
        
        # Predict future position
        steps_ahead = time_horizon / avg_time_delta
        last_x, last_y = positions[-1]
        
        predicted_x = last_x + avg_vx * steps_ahead
        predicted_y = last_y + avg_vy * steps_ahead
        
        # Calculate confidence based on velocity consistency
        vx_std = np.std(velocities_x) if len(velocities_x) > 1 else 0
        vy_std = np.std(velocities_y) if len(velocities_y) > 1 else 0
        
        velocity_magnitude = np.sqrt(avg_vx**2 + avg_vy**2)
        velocity_variation = (vx_std + vy_std) / 2
        
        if velocity_magnitude > 0:
            confidence = max(0, 1.0 - velocity_variation / velocity_magnitude)
        else:
            confidence = 0.5  # Stationary object
        
        return {
            "predicted_x": round(predicted_x, 2),
            "predicted_y": round(predicted_y, 2),
            "confidence": round(min(confidence, 1.0), 3),
            "velocity": {"x": round(avg_vx, 2), "y": round(avg_vy, 2)},
            "method": "linear"
        }
    
    def _polynomial_prediction(
        self, 
        positions: List[Tuple[float, float]], 
        time_deltas: List[float],
        time_horizon: float,
        avg_time_delta: float
    ) -> Dict[str, Any]:
        """Polynomial curve fitting prediction"""
        if len(positions) < 4:
            return self._linear_prediction(positions, time_deltas, time_horizon, avg_time_delta)
        
        # Fit polynomial to trajectory
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        t_coords = list(range(len(positions)))
        
        # Try quadratic fit
        try:
            if len(positions) >= 5:
                degree = 2
            else:
                degree = 1
            
            x_coeffs = np.polyfit(t_coords, x_coords, degree)
            y_coeffs = np.polyfit(t_coords, y_coords, degree)
            
            # Predict future position
            future_t = len(positions) + time_horizon / avg_time_delta
            
            predicted_x = np.polyval(x_coeffs, future_t)
            predicted_y = np.polyval(y_coeffs, future_t)
            
            # Calculate confidence based on fit quality
            x_pred = np.polyval(x_coeffs, t_coords)
            y_pred = np.polyval(y_coeffs, t_coords)
            
            x_error = np.mean((np.array(x_coords) - x_pred) ** 2)
            y_error = np.mean((np.array(y_coords) - y_pred) ** 2)
            
            confidence = max(0, 1.0 - (x_error + y_error) / 1000)
            
            return {
                "predicted_x": round(predicted_x, 2),
                "predicted_y": round(predicted_y, 2),
                "confidence": round(min(confidence, 1.0), 3),
                "method": "polynomial",
                "degree": degree
            }
        except Exception as e:
            logger.warning(f"Polynomial prediction failed: {e}")
            return self._linear_prediction(positions, time_deltas, time_horizon, avg_time_delta)


class ScenePredictor:
    """Predicts scene-level patterns and trends"""
    
    def __init__(self):
        """Initialize scene predictor"""
        self.scene_history: List[Dict[str, Any]] = []
        self.class_trends: Dict[str, List[Tuple[int, datetime]]] = defaultdict(list)
        
    def add_scene_observation(self, scene_data: Dict[str, Any]):
        """Add a scene observation to history"""
        self.scene_history.append({
            "timestamp": datetime.utcnow(),
            "data": scene_data
        })
        
        # Update class trends
        if "class_counts" in scene_data:
            for class_name, count in scene_data["class_counts"].items():
                self.class_trends[class_name].append((count, datetime.utcnow()))
        
        # Keep only last 100 observations
        if len(self.scene_history) > 100:
            self.scene_history = self.scene_history[-100:]
        
        for class_name in self.class_trends:
            if len(self.class_trends[class_name]) > 100:
                self.class_trends[class_name] = self.class_trends[class_name][-100:]
    
    def predict_class_emergence(self, time_horizon: int = 5) -> List[Dict[str, Any]]:
        """
        Predict which classes are likely to appear
        
        Args:
            time_horizon: Number of frames into the future
            
        Returns:
            List of predicted class emergences with confidence
        """
        predictions = []
        
        for class_name, trend_data in self.class_trends.items():
            if len(trend_data) < 5:
                continue
            
            # Analyze trend
            counts = [c for c, _ in trend_data]
            recent_counts = counts[-5:]
            
            # Calculate trend direction
            if len(recent_counts) >= 3:
                trend = np.polyfit(range(len(recent_counts)), recent_counts, 1)[0]
            else:
                trend = 0
            
            # Calculate average count
            avg_count = np.mean(recent_counts)
            
            # Predict future count
            predicted_count = avg_count + trend * time_horizon
            
            # Calculate confidence
            count_std = np.std(recent_counts)
            confidence = max(0, 1.0 - count_std / max(avg_count, 1))
            
            if predicted_count > 0:
                predictions.append({
                    "class_name": class_name,
                    "predicted_count": round(max(predicted_count, 0), 1),
                    "confidence": round(min(confidence, 1.0), 3),
                    "trend": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable",
                    "trend_magnitude": round(abs(trend), 3)
                })
        
        # Sort by confidence
        predictions.sort(key=lambda p: p["confidence"], reverse=True)
        return predictions[:10]
    
    def predict_scene_complexity(self, time_horizon: int = 5) -> Dict[str, Any]:
        """Predict future scene complexity"""
        if len(self.scene_history) < 5:
            return {
                "predicted_complexity": "unknown",
                "confidence": 0.0,
                "reasoning": ["Insufficient historical data"]
            }
        
        # Extract complexity metrics from history
        complexities = []
        for obs in self.scene_history[-10:]:
            data = obs["data"]
            if "object_count" in data:
                complexities.append(data["object_count"])
        
        if not complexities:
            return {
                "predicted_complexity": "unknown",
                "confidence": 0.0,
                "reasoning": ["No complexity data available"]
            }
        
        # Predict future complexity
        recent = complexities[-5:]
        avg_complexity = np.mean(recent)
        
        if len(recent) >= 3:
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
        else:
            trend = 0
        
        predicted = avg_complexity + trend * time_horizon
        
        # Determine complexity level
        if predicted < 3:
            level = "low"
        elif predicted < 8:
            level = "medium"
        else:
            level = "high"
        
        # Calculate confidence
        complexity_std = np.std(recent)
        confidence = max(0, 1.0 - complexity_std / max(avg_complexity, 1))
        
        return {
            "predicted_complexity": level,
            "predicted_object_count": round(max(predicted, 0), 1),
            "confidence": round(min(confidence, 1.0), 3),
            "trend": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable",
            "reasoning": [
                f"Recent average: {avg_complexity:.1f} objects",
                f"Trend: {trend:.2f} objects per frame",
                f"Variability: {complexity_std:.2f}"
            ]
        }


class PredictionEngine:
    """
    Main prediction engine that combines multiple prediction methods
    to provide comprehensive predictive analytics.
    """
    
    def __init__(self):
        """Initialize prediction engine"""
        self.trajectory_predictor = TrajectoryPredictor()
        self.scene_predictor = ScenePredictor()
        
        # Prediction statistics
        self.prediction_history: List[Dict[str, Any]] = []
        self.prediction_accuracy: Dict[str, List[bool]] = defaultdict(list)
        
    def update_with_detection(
        self, 
        detections: List[Dict[str, Any]], 
        frame_id: str,
        tracks: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Update prediction models with new detection data
        
        Args:
            detections: List of detection results
            frame_id: Frame identifier
            tracks: Optional track information
        """
        timestamp = datetime.utcnow()
        
        # Update trajectory predictions
        if tracks:
            for track_id, track_data in tracks.items():
                if "bbox" in track_data:
                    self.trajectory_predictor.update_trajectory(
                        track_id, 
                        track_data["bbox"], 
                        timestamp
                    )
        
        # Update scene predictions
        class_counts = {}
        for det in detections:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        self.scene_predictor.add_scene_observation({
            "frame_id": frame_id,
            "object_count": len(detections),
            "class_counts": class_counts,
            "unique_classes": len(class_counts)
        })
    
    def predict_object_trajectory(
        self, 
        track_id: str, 
        time_horizon: float = 1.0,
        method: str = "linear"
    ) -> Optional[Dict[str, Any]]:
        """
        Predict future position of a tracked object
        
        Args:
            track_id: Track identifier
            time_horizon: Seconds into the future
            method: Prediction method
            
        Returns:
            Prediction result
        """
        prediction = self.trajectory_predictor.predict_future_position(
            track_id, time_horizon, method
        )
        
        if prediction:
            self._record_prediction("trajectory", prediction["confidence"])
        
        return prediction
    
    def predict_scene_evolution(self, frames_ahead: int = 5) -> Dict[str, Any]:
        """
        Predict how the scene will evolve
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Scene evolution predictions
        """
        # Get class emergence predictions
        class_predictions = self.scene_predictor.predict_class_emergence(frames_ahead)
        
        # Get complexity prediction
        complexity_prediction = self.scene_predictor.predict_scene_complexity(frames_ahead)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(class_predictions, complexity_prediction)
        
        return {
            "frames_ahead": frames_ahead,
            "class_predictions": class_predictions,
            "complexity_prediction": complexity_prediction,
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_recommendations(
        self, 
        class_predictions: List[Dict[str, Any]],
        complexity_prediction: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = []
        
        # Complexity-based recommendations
        complexity = complexity_prediction.get("predicted_complexity", "unknown")
        if complexity == "high":
            recommendations.append("Consider using a more accurate model for complex scene")
            recommendations.append("Increase confidence threshold to reduce false positives")
        elif complexity == "low":
            recommendations.append("Fast model should be sufficient for simple scene")
            recommendations.append("Can lower confidence threshold for more detections")
        
        # Class-based recommendations
        emerging_classes = [p for p in class_predictions if p["trend"] == "increasing" and p["confidence"] > 0.7]
        if emerging_classes:
            classes = ", ".join([p["class_name"] for p in emerging_classes])
            recommendations.append(f"Watch for emerging objects: {classes}")
        
        disappearing_classes = [p for p in class_predictions if p["trend"] == "decreasing" and p["confidence"] > 0.7]
        if disappearing_classes:
            classes = ", ".join([p["class_name"] for p in disappearing_classes])
            recommendations.append(f"Objects may be leaving scene: {classes}")
        
        return recommendations
    
    def get_next_frame_predictions(self) -> Dict[str, Any]:
        """Get predictions for the next frame"""
        return self.predict_scene_evolution(1)
    
    def _record_prediction(self, prediction_type: str, confidence: float):
        """Record a prediction for accuracy tracking"""
        self.prediction_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": prediction_type,
            "confidence": confidence
        })
        
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def record_prediction_outcome(self, prediction_type: str, was_accurate: bool):
        """Record whether a prediction was accurate"""
        self.prediction_accuracy[prediction_type].append(was_accurate)
        
        # Keep only last 100 outcomes per type
        if len(self.prediction_accuracy[prediction_type]) > 100:
            self.prediction_accuracy[prediction_type] = self.prediction_accuracy[prediction_type][-100:]
    
    def get_prediction_accuracy_stats(self) -> Dict[str, Any]:
        """Get prediction accuracy statistics"""
        stats = {}
        for prediction_type, outcomes in self.prediction_accuracy.items():
            if outcomes:
                stats[prediction_type] = {
                    "total_predictions": len(outcomes),
                    "accurate_predictions": sum(outcomes),
                    "accuracy_rate": round(sum(outcomes) / len(outcomes), 3)
                }
        return stats
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get comprehensive prediction summary"""
        return {
            "total_predictions": len(self.prediction_history),
            "accuracy_stats": self.get_prediction_accuracy_stats(),
            "recent_predictions": self.prediction_history[-10:],
            "scene_trends": self.scene_predictor.predict_class_emergence(5),
            "complexity_trend": self.scene_predictor.predict_scene_complexity(5)
        }
    
    def clear(self):
        """Clear all prediction data"""
        self.trajectory_predictor.trajectory_history.clear()
        self.scene_predictor.scene_history.clear()
        self.scene_predictor.class_trends.clear()
        self.prediction_history.clear()
        self.prediction_accuracy.clear()


# Singleton instance
_prediction_engine: Optional[PredictionEngine] = None


def get_prediction_engine() -> PredictionEngine:
    """Get or create prediction engine singleton"""
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = PredictionEngine()
    return _prediction_engine