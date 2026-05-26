"""
Scene Memory Timeline Service
Maintains a temporal history of detected scenes and objects,
enabling tracking of changes over time and pattern recognition.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)


class SceneFrame:
    """Represents a single frame/scene in the timeline"""
    
    def __init__(
        self,
        frame_id: str,
        timestamp: datetime,
        detections: List[Dict[str, Any]],
        image_dimensions: Dict[str, int],
        model_used: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.detections = detections
        self.image_dimensions = image_dimensions
        self.model_used = model_used
        self.session_id = session_id
        self.metadata = metadata or {}
        
        # Computed properties
        self.object_count = len(detections)
        self.unique_classes = list(set(d["class_name"] for d in detections))
        self.class_counts = self._compute_class_counts()
        
    def _compute_class_counts(self) -> Dict[str, int]:
        """Compute count of each class in detections"""
        counts = {}
        for det in self.detections:
            class_name = det["class_name"]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to dictionary"""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "detections": self.detections,
            "image_dimensions": self.image_dimensions,
            "object_count": self.object_count,
            "unique_classes": self.unique_classes,
            "class_counts": self.class_counts,
            "model_used": self.model_used,
            "session_id": self.session_id,
            "metadata": self.metadata
        }


class ObjectTrack:
    """Represents a tracked object across multiple frames"""
    
    def __init__(self, track_id: str, class_name: str, first_seen: datetime):
        self.track_id = track_id
        self.class_name = class_name
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.observations: List[Dict[str, Any]] = []
        self.bbox_history: List[Dict[str, float]] = []
        self.confidence_history: List[float] = []
        self.frame_ids: List[str] = []
        
    def add_observation(
        self, 
        frame_id: str, 
        bbox: Dict[str, float], 
        confidence: float,
        timestamp: datetime
    ):
        """Add an observation to the track"""
        self.last_seen = timestamp
        self.frame_ids.append(frame_id)
        self.bbox_history.append(bbox)
        self.confidence_history.append(confidence)
        
        self.observations.append({
            "frame_id": frame_id,
            "timestamp": timestamp.isoformat(),
            "bbox": bbox,
            "confidence": confidence
        })
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Get center point trajectory"""
        trajectory = []
        for bbox in self.bbox_history:
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2
            trajectory.append((center_x, center_y))
        return trajectory
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get track statistics"""
        if not self.confidence_history:
            return {}
            
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "duration_seconds": (self.last_seen - self.first_seen).total_seconds(),
            "observation_count": len(self.observations),
            "avg_confidence": round(np.mean(self.confidence_history), 3),
            "max_confidence": round(max(self.confidence_history), 3),
            "min_confidence": round(min(self.confidence_history), 3),
            "trajectory_points": len(self.get_trajectory())
        }


class SceneMemoryTimeline:
    """
    Maintains a temporal timeline of scenes and object detections,
    enabling tracking of changes, patterns, and object persistence.
    """
    
    def __init__(self, max_frames: int = 1000, max_age_seconds: float = 3600):
        """
        Initialize scene memory timeline
        
        Args:
            max_frames: Maximum number of frames to keep in memory
            max_age_seconds: Maximum age of frames to keep (1 hour default)
        """
        self.max_frames = max_frames
        self.max_age_seconds = max_age_seconds
        
        # Frame storage
        self.frames: deque = deque(maxlen=max_frames)
        self.frames_by_id: Dict[str, SceneFrame] = {}
        self.frames_by_session: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_frames))
        
        # Object tracking
        self.object_tracks: Dict[str, ObjectTrack] = {}
        self.next_track_id = 0
        
        # Scene statistics
        self.scene_statistics = {
            "total_frames": 0,
            "total_detections": 0,
            "total_objects_tracked": 0,
            "class_frequency": defaultdict(int),
            "session_count": 0,
            "active_sessions": set()
        }
        
        # Temporal patterns
        self.temporal_patterns: List[Dict[str, Any]] = []
        
    def add_frame(
        self,
        detections: List[Dict[str, Any]],
        image_dimensions: Dict[str, int],
        model_used: str,
        session_id: Optional[str] = None,
        frame_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SceneFrame:
        """
        Add a new frame to the timeline
        
        Args:
            detections: List of detection results
            image_dimensions: Image width and height
            model_used: Model identifier used for detection
            session_id: Optional session identifier
            frame_id: Optional frame identifier (generated if not provided)
            metadata: Optional additional metadata
            
        Returns:
            SceneFrame object
        """
        timestamp = datetime.utcnow()
        
        if frame_id is None:
            frame_id = f"frame_{self.scene_statistics['total_frames'] + 1}_{int(timestamp.timestamp() * 1000)}"
        
        # Create scene frame
        frame = SceneFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            detections=detections,
            image_dimensions=image_dimensions,
            model_used=model_used,
            session_id=session_id,
            metadata=metadata
        )
        
        # Store frame
        self.frames.append(frame)
        self.frames_by_id[frame_id] = frame
        
        if session_id:
            self.frames_by_session[session_id].append(frame)
            self.scene_statistics["active_sessions"].add(session_id)
        
        # Update statistics
        self.scene_statistics["total_frames"] += 1
        self.scene_statistics["total_detections"] += len(detections)
        
        for det in detections:
            self.scene_statistics["class_frequency"][det["class_name"]] += 1
        
        # Track objects across frames
        self._track_objects(frame)
        
        # Detect temporal patterns
        self._detect_temporal_patterns()
        
        # Clean old frames
        self._cleanup_old_frames()
        
        return frame
    
    def _track_objects(self, frame: SceneFrame):
        """Track objects across frames using simple spatial matching"""
        if not self.object_tracks:
            # First frame - create tracks for all detections
            for i, det in enumerate(frame.detections):
                track_id = f"track_{self.next_track_id}"
                self.next_track_id += 1
                
                track = ObjectTrack(
                    track_id=track_id,
                    class_name=det["class_name"],
                    first_seen=frame.timestamp
                )
                
                track.add_observation(
                    frame_id=frame.frame_id,
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    timestamp=frame.timestamp
                )
                
                self.object_tracks[track_id] = track
                self.scene_statistics["total_objects_tracked"] += 1
        else:
            # Match detections to existing tracks
            used_tracks = set()
            
            for det in frame.detections:
                best_track = None
                best_iou = 0
                
                # Find best matching track
                for track_id, track in self.object_tracks.items():
                    if track_id in used_tracks:
                        continue
                    
                    if track.class_name != det["class_name"]:
                        continue
                    
                    # Calculate IoU with last known position
                    if track.bbox_history:
                        last_bbox = track.bbox_history[-1]
                        iou = self._calculate_iou(last_bbox, det["bbox"])
                        
                        if iou > best_iou and iou > 0.3:  # Threshold for matching
                            best_iou = iou
                            best_track = track_id
                
                if best_track:
                    # Update existing track
                    self.object_tracks[best_track].add_observation(
                        frame_id=frame.frame_id,
                        bbox=det["bbox"],
                        confidence=det["confidence"],
                        timestamp=frame.timestamp
                    )
                    used_tracks.add(best_track)
                else:
                    # Create new track
                    track_id = f"track_{self.next_track_id}"
                    self.next_track_id += 1
                    
                    track = ObjectTrack(
                        track_id=track_id,
                        class_name=det["class_name"],
                        first_seen=frame.timestamp
                    )
                    
                    track.add_observation(
                        frame_id=frame.frame_id,
                        bbox=det["bbox"],
                        confidence=det["confidence"],
                        timestamp=frame.timestamp
                    )
                    
                    self.object_tracks[track_id] = track
                    self.scene_statistics["total_objects_tracked"] += 1
    
    def _calculate_iou(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(bbox1["x1"], bbox2["x1"])
        y1 = max(bbox1["y1"], bbox2["y1"])
        x2 = min(bbox1["x2"], bbox2["x2"])
        y2 = min(bbox1["y2"], bbox2["y2"])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1["x2"] - bbox1["x1"]) * (bbox1["y2"] - bbox1["y1"])
        area2 = (bbox2["x2"] - bbox2["x1"]) * (bbox2["y2"] - bbox2["y1"])
        
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0
        
        return intersection / union
    
    def _detect_temporal_patterns(self):
        """Detect patterns in the temporal data"""
        if len(self.frames) < 5:
            return
        
        # Analyze last 5 frames for patterns
        recent_frames = list(self.frames)[-5:]
        
        # Check for increasing/decreasing object counts
        object_counts = [f.object_count for f in recent_frames]
        count_trend = "stable"
        if all(object_counts[i] < object_counts[i+1] for i in range(len(object_counts)-1)):
            count_trend = "increasing"
        elif all(object_counts[i] > object_counts[i+1] for i in range(len(object_counts)-1)):
            count_trend = "decreasing"
        
        # Check for class consistency
        recent_classes = set()
        for frame in recent_frames:
            recent_classes.update(frame.unique_classes)
        
        consistency_ratio = len(recent_classes) / max(len([c for f in recent_frames for c in f.unique_classes]), 1)
        
        pattern = {
            "timestamp": datetime.utcnow().isoformat(),
            "frame_range": [recent_frames[0].frame_id, recent_frames[-1].frame_id],
            "object_count_trend": count_trend,
            "class_consistency": round(consistency_ratio, 3),
            "unique_classes_observed": len(recent_classes),
            "avg_objects_per_frame": round(np.mean(object_counts), 2)
        }
        
        self.temporal_patterns.append(pattern)
        
        # Keep only last 100 patterns
        if len(self.temporal_patterns) > 100:
            self.temporal_patterns = self.temporal_patterns[-100:]
    
    def _cleanup_old_frames(self):
        """Remove frames older than max_age_seconds"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.max_age_seconds)
        
        # Clean main frames deque
        while self.frames and self.frames[0].timestamp < cutoff_time:
            old_frame = self.frames.popleft()
            if old_frame.frame_id in self.frames_by_id:
                del self.frames_by_id[old_frame.frame_id]
        
        # Clean session-specific frames
        for session_id, session_frames in list(self.frames_by_session.items()):
            while session_frames and session_frames[0].timestamp < cutoff_time:
                session_frames.popleft()
            
            # Remove empty sessions
            if not session_frames:
                del self.frames_by_session[session_id]
                if session_id in self.scene_statistics["active_sessions"]:
                    self.scene_statistics["active_sessions"].discard(session_id)
    
    def get_timeline(
        self, 
        limit: int = 50, 
        session_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of frames
        
        Args:
            limit: Maximum number of frames to return
            session_id: Filter by session
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List of frame dictionaries
        """
        if session_id and session_id in self.frames_by_session:
            frames = list(self.frames_by_session[session_id])
        else:
            frames = list(self.frames)
        
        # Apply time filters
        if start_time:
            frames = [f for f in frames if f.timestamp >= start_time]
        if end_time:
            frames = [f for f in frames if f.timestamp <= end_time]
        
        # Sort by timestamp (newest first) and limit
        frames.sort(key=lambda f: f.timestamp, reverse=True)
        frames = frames[:limit]
        
        return [f.to_dict() for f in frames]
    
    def get_object_tracks(
        self, 
        class_filter: Optional[str] = None,
        min_observations: int = 1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get object tracks
        
        Args:
            class_filter: Filter by class name
            min_observations: Minimum number of observations
            limit: Maximum number of tracks to return
            
        Returns:
            List of track statistics
        """
        tracks = []
        for track in self.object_tracks.values():
            if class_filter and track.class_name != class_filter:
                continue
            if len(track.observations) < min_observations:
                continue
            
            tracks.append(track.get_statistics())
        
        # Sort by observation count (most observed first)
        tracks.sort(key=lambda t: t["observation_count"], reverse=True)
        return tracks[:limit]
    
    def get_track_trajectory(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed trajectory for a specific track"""
        if track_id not in self.object_tracks:
            return None
        
        track = self.object_tracks[track_id]
        trajectory = track.get_trajectory()
        
        return {
            "track_id": track_id,
            "class_name": track.class_name,
            "trajectory": [{"x": p[0], "y": p[1]} for p in trajectory],
            "observations": track.observations,
            "statistics": track.get_statistics()
        }
    
    def get_scene_statistics(self) -> Dict[str, Any]:
        """Get overall scene statistics"""
        return {
            "total_frames": self.scene_statistics["total_frames"],
            "total_detections": self.scene_statistics["total_detections"],
            "total_objects_tracked": self.scene_statistics["total_objects_tracked"],
            "class_frequency": dict(self.scene_statistics["class_frequency"]),
            "session_count": len(self.scene_statistics["active_sessions"]),
            "active_sessions": list(self.scene_statistics["active_sessions"]),
            "avg_detections_per_frame": round(
                self.scene_statistics["total_detections"] / max(self.scene_statistics["total_frames"], 1), 
                2
            ),
            "temporal_patterns": self.temporal_patterns[-10:] if self.temporal_patterns else []
        }
    
    def get_class_timeline(self, class_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get timeline of detections for a specific class"""
        timeline = []
        
        for frame in list(self.frames)[-limit:]:
            class_detections = [d for d in frame.detections if d["class_name"] == class_name]
            if class_detections:
                timeline.append({
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp.isoformat(),
                    "count": len(class_detections),
                    "detections": class_detections
                })
        
        return timeline
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific session"""
        if session_id not in self.frames_by_session:
            return None
        
        session_frames = list(self.frames_by_session[session_id])
        if not session_frames:
            return None
        
        # Calculate session statistics
        total_detections = sum(f.object_count for f in session_frames)
        all_classes = set()
        class_counts = defaultdict(int)
        
        for frame in session_frames:
            all_classes.update(frame.unique_classes)
            for class_name, count in frame.class_counts.items():
                class_counts[class_name] += count
        
        return {
            "session_id": session_id,
            "frame_count": len(session_frames),
            "total_detections": total_detections,
            "unique_classes": list(all_classes),
            "class_counts": dict(class_counts),
            "start_time": session_frames[0].timestamp.isoformat(),
            "end_time": session_frames[-1].timestamp.isoformat(),
            "duration_seconds": (session_frames[-1].timestamp - session_frames[0].timestamp).total_seconds(),
            "avg_detections_per_frame": round(total_detections / len(session_frames), 2)
        }
    
    def clear(self):
        """Clear all data"""
        self.frames.clear()
        self.frames_by_id.clear()
        self.frames_by_session.clear()
        self.object_tracks.clear()
        self.next_track_id = 0
        self.scene_statistics = {
            "total_frames": 0,
            "total_detections": 0,
            "total_objects_tracked": 0,
            "class_frequency": defaultdict(int),
            "session_count": 0,
            "active_sessions": set()
        }
        self.temporal_patterns.clear()


# Singleton instance
_scene_memory: Optional[SceneMemoryTimeline] = None


def get_scene_memory() -> SceneMemoryTimeline:
    """Get or create scene memory timeline singleton"""
    global _scene_memory
    if _scene_memory is None:
        _scene_memory = SceneMemoryTimeline()
    return _scene_memory