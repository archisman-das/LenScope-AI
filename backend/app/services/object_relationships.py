"""
Object Relationship Graph Service
Analyzes and maintains relationships between detected objects
based on spatial proximity, co-occurrence patterns, and semantic relationships.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class ObjectNode:
    """Represents a node in the object relationship graph"""
    
    def __init__(self, node_id: str, class_name: str, bbox: Dict[str, float], confidence: float):
        self.node_id = node_id
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.center = self._calculate_center()
        self.area = self._calculate_area()
        self.attributes: Dict[str, Any] = {}
        self.relationships: List["RelationshipEdge"] = []
        
    def _calculate_center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box"""
        return (
            (self.bbox["x1"] + self.bbox["x2"]) / 2,
            (self.bbox["y1"] + self.bbox["y2"]) / 2
        )
    
    def _calculate_area(self) -> float:
        """Calculate area of bounding box"""
        return (self.bbox["x2"] - self.bbox["x1"]) * (self.bbox["y2"] - self.bbox["y1"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            "node_id": self.node_id,
            "class_name": self.class_name,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "center": {"x": self.center[0], "y": self.center[1]},
            "area": self.area,
            "attributes": self.attributes,
            "relationship_count": len(self.relationships)
        }


class RelationshipEdge:
    """Represents a relationship between two object nodes"""
    
    def __init__(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: str, 
        strength: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.strength = strength  # 0.0 to 1.0
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
    def update(self, strength: float, metadata: Optional[Dict[str, Any]] = None):
        """Update edge properties"""
        self.strength = strength
        if metadata:
            self.metadata.update(metadata)
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "strength": round(self.strength, 3),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class ObjectRelationshipGraph:
    """
    Maintains a graph of object relationships based on:
    - Spatial proximity
    - Co-occurrence patterns
    - Semantic relationships
    - Interaction patterns
    """
    
    def __init__(self):
        """Initialize object relationship graph"""
        self.nodes: Dict[str, ObjectNode] = {}
        self.edges: Dict[str, RelationshipEdge] = {}
        self.next_node_id = 0
        
        # Relationship statistics
        self.relationship_statistics = {
            "total_nodes": 0,
            "total_edges": 0,
            "relationship_types": defaultdict(int),
            "class_co_occurrences": defaultdict(lambda: defaultdict(int)),
            "spatial_patterns": defaultdict(int)
        }
        
        # Semantic relationship definitions
        self.semantic_relationships = self._initialize_semantic_relationships()
        
        # Historical data for pattern analysis
        self.co_occurrence_history: List[Dict[str, Any]] = []
        
    def _initialize_semantic_relationships(self) -> Dict[str, List[str]]:
        """Initialize common semantic relationships between object classes"""
        return {
            # Human-related relationships
            "person": ["chair", "couch", "bed", "dining table", "laptop", "cell phone", "book", "tv", "remote"],
            
            # Furniture relationships
            "chair": ["dining table", "person", "couch"],
            "couch": ["person", "tv", "dining table", "potted plant"],
            "bed": ["person", "book", "teddy bear"],
            "dining table": ["chair", "person", "cup", "bowl", "fork", "knife", "spoon", "bottle", "wine glass", "pizza", "cake", "donut"],
            
            # Kitchen relationships
            "refrigerator": ["bottle", "cup", "bowl", "banana", "apple", "orange", "carrot", "broccoli"],
            "microwave": ["cup", "bowl", "pizza", "donut"],
            "oven": ["bowl", "pizza", "cake", "donut"],
            "toaster": ["bread", "donut"],
            "sink": ["cup", "bowl", "fork", "knife", "spoon", "bottle"],
            
            # Office relationships
            "laptop": ["person", "mouse", "keyboard", "cell phone", "cup", "book"],
            "keyboard": ["mouse", "laptop", "person"],
            "mouse": ["keyboard", "laptop", "person"],
            "book": ["person", "laptop", "chair", "bed"],
            
            # Entertainment relationships
            "tv": ["person", "couch", "remote", "chair"],
            "remote": ["tv", "person", "couch"],
            
            # Outdoor relationships
            "car": ["person", "truck", "traffic light", "stop sign", "parking meter"],
            "bicycle": ["person", "backpack", "handbag"],
            "motorcycle": ["person", "car", "truck"],
            
            # Animal relationships
            "dog": ["person", "cat", "frisbee", "sports ball"],
            "cat": ["person", "dog", "couch", "bed"],
            "horse": ["person", "zebra", "giraffe"],
            
            # Fashion relationships
            "handbag": ["person", "suitcase", "umbrella"],
            "backpack": ["person", "suitcase", "bicycle"],
            "umbrella": ["person", "handbag"],
            "tie": ["person", "suitcase"],
            
            # Food relationships
            "banana": ["apple", "orange", "bowl"],
            "apple": ["banana", "orange", "bowl"],
            "pizza": ["dining table", "cup", "bottle", "person"],
            "cake": ["dining table", "person", "cup"],
            "donut": ["coffee", "cup", "dining table", "person"],
        }
    
    def add_detections(
        self, 
        detections: List[Dict[str, Any]], 
        frame_id: str, 
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add detections to the graph and update relationships
        
        Args:
            detections: List of detection results
            frame_id: Frame identifier
            session_id: Optional session identifier
            
        Returns:
            Dictionary with graph update information
        """
        timestamp = datetime.utcnow()
        
        # Create nodes for new detections
        new_nodes = []
        for det in detections:
            node_id = f"node_{self.next_node_id}_{frame_id}"
            self.next_node_id += 1
            
            node = ObjectNode(
                node_id=node_id,
                class_name=det["class_name"],
                bbox=det["bbox"],
                confidence=det["confidence"]
            )
            
            # Add attributes
            node.attributes["frame_id"] = frame_id
            node.attributes["session_id"] = session_id
            node.attributes["created_at"] = timestamp.isoformat()
            
            self.nodes[node_id] = node
            new_nodes.append(node)
        
        # Update relationship statistics
        self.relationship_statistics["total_nodes"] += len(detections)
        
        # Track co-occurrences
        classes_in_frame = [det["class_name"] for det in detections]
        self._update_co_occurrences(classes_in_frame, frame_id, timestamp)
        
        # Create/update relationships
        relationships_created = self._create_relationships(new_nodes, detections)
        
        # Analyze spatial patterns
        spatial_patterns = self._analyze_spatial_patterns(new_nodes)
        
        return {
            "frame_id": frame_id,
            "nodes_added": len(new_nodes),
            "relationships_created": relationships_created,
            "spatial_patterns": spatial_patterns,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges)
        }
    
    def _update_co_occurrences(self, classes: List[str], frame_id: str, timestamp: datetime):
        """Update co-occurrence statistics"""
        unique_classes = list(set(classes))
        
        for i, class1 in enumerate(unique_classes):
            for class2 in unique_classes[i+1:]:
                self.relationship_statistics["class_co_occurrences"][class1][class2] += 1
                self.relationship_statistics["class_co_occurrences"][class2][class1] += 1
        
        # Record in history
        self.co_occurrence_history.append({
            "frame_id": frame_id,
            "timestamp": timestamp.isoformat(),
            "classes": unique_classes,
            "co_occurrences": {
                f"{c1}-{c2}": self.relationship_statistics["class_co_occurrences"][c1][c2]
                for i, c1 in enumerate(unique_classes)
                for c2 in unique_classes[i+1:]
            }
        })
        
        # Keep only last 1000 records
        if len(self.co_occurrence_history) > 1000:
            self.co_occurrence_history = self.co_occurrence_history[-1000:]
    
    def _create_relationships(self, new_nodes: List[ObjectNode], detections: List[Dict[str, Any]]) -> int:
        """Create relationships between nodes"""
        relationships_created = 0
        
        # Get all existing nodes
        all_nodes = list(self.nodes.values())
        
        for new_node in new_nodes:
            # Check relationships with all other nodes
            for existing_node in all_nodes:
                if existing_node.node_id == new_node.node_id:
                    continue
                
                # Calculate spatial relationship
                spatial_relationship = self._calculate_spatial_relationship(new_node, existing_node)
                
                if spatial_relationship:
                    edge_id = f"edge_{new_node.node_id}_{existing_node.node_id}"
                    
                    if edge_id not in self.edges:
                        edge = RelationshipEdge(
                            source_id=new_node.node_id,
                            target_id=existing_node.node_id,
                            relationship_type=spatial_relationship["type"],
                            strength=spatial_relationship["strength"],
                            metadata={
                                "distance": spatial_relationship["distance"],
                                "iou": spatial_relationship["iou"],
                                "is_semantic": spatial_relationship["is_semantic"]
                            }
                        )
                        
                        self.edges[edge_id] = edge
                        new_node.relationships.append(edge)
                        existing_node.relationships.append(edge)
                        
                        self.relationship_statistics["total_edges"] += 1
                        self.relationship_statistics["relationship_types"][spatial_relationship["type"]] += 1
                        
                        relationships_created += 1
                    else:
                        # Update existing edge
                        edge = self.edges[edge_id]
                        edge.update(
                            strength=spatial_relationship["strength"],
                            metadata={
                                **edge.metadata,
                                "distance": spatial_relationship["distance"],
                                "iou": spatial_relationship["iou"],
                                "last_updated": datetime.utcnow().isoformat()
                            }
                        )
        
        return relationships_created
    
    def _calculate_spatial_relationship(
        self, 
        node1: ObjectNode, 
        node2: ObjectNode
    ) -> Optional[Dict[str, Any]]:
        """Calculate spatial relationship between two nodes"""
        # Calculate distance between centers
        distance = np.sqrt(
            (node1.center[0] - node2.center[0]) ** 2 + 
            (node1.center[1] - node2.center[1]) ** 2
        )
        
        # Calculate IoU
        iou = self._calculate_iou(node1.bbox, node2.bbox)
        
        # Check semantic relationship
        is_semantic = self._check_semantic_relationship(node1.class_name, node2.class_name)
        
        # Determine relationship type and strength
        relationship_type = None
        strength = 0.0
        
        # High IoU indicates overlap/containment
        if iou > 0.5:
            relationship_type = "overlap"
            strength = min(iou, 1.0)
        # Close proximity
        elif distance < 50:
            relationship_type = "near"
            strength = max(0, 1.0 - distance / 50)
        # Medium proximity
        elif distance < 150:
            relationship_type = "proximity"
            strength = max(0, 0.7 - distance / 300)
        # Same region (loose association)
        elif distance < 300:
            if is_semantic:
                relationship_type = "semantic_association"
                strength = 0.3
            else:
                return None  # No significant relationship
        else:
            if is_semantic and node1.area > 10000 and node2.area > 10000:
                relationship_type = "weak_semantic"
                strength = 0.1
            else:
                return None
        
        # Boost strength for semantic relationships
        if is_semantic:
            strength = min(strength * 1.2, 1.0)
        
        return {
            "type": relationship_type,
            "strength": round(strength, 3),
            "distance": round(distance, 2),
            "iou": round(iou, 4),
            "is_semantic": is_semantic
        }
    
    def _calculate_iou(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """Calculate Intersection over Union"""
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
    
    def _check_semantic_relationship(self, class1: str, class2: str) -> bool:
        """Check if two classes have a semantic relationship"""
        if class1 in self.semantic_relationships:
            return class2 in self.semantic_relationships[class1]
        if class2 in self.semantic_relationships:
            return class1 in self.semantic_relationships[class2]
        return False
    
    def _analyze_spatial_patterns(self, nodes: List[ObjectNode]) -> Dict[str, Any]:
        """Analyze spatial patterns in the scene"""
        patterns = {
            "clusters": 0,
            "isolated_objects": 0,
            "dominant_relationships": [],
            "spatial_distribution": "unknown"
        }
        
        if not nodes:
            return patterns
        
        # Count isolated objects (no relationships)
        for node in nodes:
            if not node.relationships:
                patterns["isolated_objects"] += 1
        
        # Find dominant relationship types
        relationship_counts = defaultdict(int)
        for edge in self.edges.values():
            relationship_counts[edge.relationship_type] += 1
        
        if relationship_counts:
            patterns["dominant_relationships"] = sorted(
                relationship_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
        
        # Analyze spatial distribution
        if len(nodes) >= 3:
            centers = [node.center for node in nodes]
            x_coords = [c[0] for c in centers]
            y_coords = [c[1] for c in centers]
            
            x_std = np.std(x_coords)
            y_std = np.std(y_coords)
            
            if x_std < 100 and y_std < 100:
                patterns["spatial_distribution"] = "clustered"
            elif x_std > 300 or y_std > 300:
                patterns["spatial_distribution"] = "dispersed"
            else:
                patterns["spatial_distribution"] = "distributed"
        
        return patterns
    
    def get_graph(self, include_edges: bool = True) -> Dict[str, Any]:
        """Get the complete graph structure"""
        graph = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges)
        }
        
        if include_edges:
            graph["edges"] = [edge.to_dict() for edge in self.edges.values()]
        
        return graph
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific node"""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        return {
            **node.to_dict(),
            "relationships": [
                {
                    "target_id": edge.target_id if edge.source_id == node_id else edge.source_id,
                    "relationship_type": edge.relationship_type,
                    "strength": edge.strength,
                    "metadata": edge.metadata
                }
                for edge in node.relationships
            ]
        }
    
    def get_relationships_for_class(self, class_name: str) -> Dict[str, Any]:
        """Get all relationships involving a specific class"""
        class_nodes = [n for n in self.nodes.values() if n.class_name == class_name]
        class_edges = []
        
        for edge in self.edges.values():
            source_node = self.nodes.get(edge.source_id)
            target_node = self.nodes.get(edge.target_id)
            
            if source_node and target_node:
                if source_node.class_name == class_name or target_node.class_name == class_name:
                    class_edges.append(edge.to_dict())
        
        return {
            "class_name": class_name,
            "node_count": len(class_nodes),
            "relationship_count": len(class_edges),
            "relationships": class_edges,
            "most_common_relationships": self._get_most_common_relationships(class_edges)
        }
    
    def _get_most_common_relationships(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get most common relationship types from edges"""
        type_counts = defaultdict(int)
        for edge in edges:
            type_counts[edge["relationship_type"]] += 1
        
        return sorted(
            [{"type": k, "count": v} for k, v in type_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:5]
    
    def get_co_occurrence_matrix(self) -> Dict[str, Dict[str, int]]:
        """Get co-occurrence matrix for all classes"""
        return {
            class1: dict(classes) 
            for class1, classes in self.relationship_statistics["class_co_occurrences"].items()
        }
    
    def get_relationship_statistics(self) -> Dict[str, Any]:
        """Get overall relationship statistics"""
        return {
            "total_nodes": self.relationship_statistics["total_nodes"],
            "total_edges": self.relationship_statistics["total_edges"],
            "relationship_types": dict(self.relationship_statistics["relationship_types"]),
            "top_co_occurrences": self._get_top_co_occurrences(10),
            "graph_density": self._calculate_graph_density()
        }
    
    def _get_top_co_occurrences(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top co-occurring class pairs"""
        pairs = []
        seen = set()
        
        for class1, classes in self.relationship_statistics["class_co_occurrences"].items():
            for class2, count in classes.items():
                pair_key = tuple(sorted([class1, class2]))
                if pair_key not in seen:
                    pairs.append({
                        "class1": class1,
                        "class2": class2,
                        "count": count
                    })
                    seen.add(pair_key)
        
        pairs.sort(key=lambda x: x["count"], reverse=True)
        return pairs[:limit]
    
    def _calculate_graph_density(self) -> float:
        """Calculate graph density"""
        n = len(self.nodes)
        if n < 2:
            return 0.0
        
        max_edges = n * (n - 1) / 2
        actual_edges = len(self.edges)
        
        return round(actual_edges / max_edges, 4) if max_edges > 0 else 0.0
    
    def get_subgraph_for_frame(self, frame_id: str) -> Dict[str, Any]:
        """Get subgraph for a specific frame"""
        frame_nodes = [n for n in self.nodes.values() if n.attributes.get("frame_id") == frame_id]
        frame_node_ids = {n.node_id for n in frame_nodes}
        
        frame_edges = [
            e for e in self.edges.values()
            if e.source_id in frame_node_ids and e.target_id in frame_node_ids
        ]
        
        return {
            "frame_id": frame_id,
            "nodes": [n.to_dict() for n in frame_nodes],
            "edges": [e.to_dict() for e in frame_edges],
            "node_count": len(frame_nodes),
            "edge_count": len(frame_edges)
        }
    
    def clear(self):
        """Clear the entire graph"""
        self.nodes.clear()
        self.edges.clear()
        self.next_node_id = 0
        self.relationship_statistics = {
            "total_nodes": 0,
            "total_edges": 0,
            "relationship_types": defaultdict(int),
            "class_co_occurrences": defaultdict(lambda: defaultdict(int)),
            "spatial_patterns": defaultdict(int)
        }
        self.co_occurrence_history.clear()


# Singleton instance
_object_graph: Optional[ObjectRelationshipGraph] = None


def get_object_graph() -> ObjectRelationshipGraph:
    """Get or create object relationship graph singleton"""
    global _object_graph
    if _object_graph is None:
        _object_graph = ObjectRelationshipGraph()
    return _object_graph