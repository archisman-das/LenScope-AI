"""
Helper utility functions
"""

import uuid
import time
import hashlib
from datetime import datetime
from typing import Optional


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())


def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix"""
    unique_id = uuid.uuid4().hex[:12]
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human-readable string"""
    if size_bytes < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hr"


def hash_password(password: str, salt: Optional[str] = None) -> tuple:
    """
    Hash a password with a salt
    
    Returns:
        (hashed_password, salt)
    """
    if salt is None:
        salt = uuid.uuid4().hex
    
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return hashed, salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """Verify a password against a hash"""
    new_hash, _ = hash_password(password, salt)
    return new_hash == hashed_password


def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO format timestamp string to datetime"""
    return datetime.fromisoformat(timestamp_str)


def calculate_fps(start_time: float, frame_count: int) -> float:
    """Calculate frames per second"""
    elapsed = time.time() - start_time
    if elapsed == 0:
        return 0
    return frame_count / elapsed


class Timer:
    """Simple timer utility for performance measurement"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time"""
        self.end_time = time.perf_counter()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return self.elapsed() * 1000
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range"""
    if max_val == min_val:
        return 0
    return (value - min_val) / (max_val - min_val)


def denormalize(value: float, min_val: float, max_val: float) -> float:
    """Denormalize value from 0-1 range"""
    return min_val + value * (max_val - min_val)


def iou(box1: list, box2: list) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate box areas
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    if union_area == 0:
        return 0
    return intersection_area / union_area