"""
Image processing utilities for detection visualization and manipulation
"""

import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import uuid
import logging

from ..config import settings, COCO_CLASSES

logger = logging.getLogger(__name__)

# Color palette for bounding boxes (neon-style colors for futuristic theme)
COLORS = [
    (0, 255, 255),    # Cyan
    (0, 255, 0),      # Green
    (255, 0, 255),    # Magenta
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Yellow
    (128, 0, 255),    # Orange
    (255, 128, 0),    # Pink
    (0, 128, 255),    # Light Blue
    (128, 255, 0),    # Lime
]


def load_image(source: Any) -> np.ndarray:
    """
    Load image from various sources (file path, URL, numpy array, bytes)
    Supports a wide range of image formats including JPG, PNG, BMP, WebP, GIF, TIFF, etc.
    
    Args:
        source: File path, URL, numpy array, or bytes
        
    Returns:
        numpy array in BGR format (OpenCV standard)
    """
    if isinstance(source, np.ndarray):
        return source
    
    if isinstance(source, bytes):
        # Try OpenCV first
        nparr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        
        # Fallback to PIL for formats OpenCV might not handle
        try:
            from PIL import Image as PILImage
            import io
            
            img_pil = PILImage.open(io.BytesIO(source))
            img_pil.load()
            
            # Convert to RGB if necessary
            if img_pil.mode not in ('RGB', 'L'):
                if img_pil.mode == 'P':
                    img_pil = img_pil.convert('RGB')
                elif img_pil.mode == 'RGBA':
                    background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                    background.paste(img_pil, mask=img_pil.split()[3])
                    img_pil = background
                elif img_pil.mode == 'LA':
                    background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                    background.paste(img_pil.convert('RGB'), mask=img_pil.split()[1])
                    img_pil = background
                else:
                    img_pil = img_pil.convert('RGB')
            
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Could not decode image from bytes: {str(e)}")
    
    if isinstance(source, (str, Path)):
        source_str = str(source)
        
        # Check if it's a URL
        if source_str.startswith(('http://', 'https://')):
            import requests
            response = requests.get(source_str)
            response.raise_for_status()
            
            # Try OpenCV first
            nparr = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            
            # Fallback to PIL
            try:
                from PIL import Image as PILImage
                import io
                
                img_pil = PILImage.open(io.BytesIO(response.content))
                img_pil.load()
                
                if img_pil.mode not in ('RGB', 'L'):
                    if img_pil.mode == 'P':
                        img_pil = img_pil.convert('RGB')
                    elif img_pil.mode == 'RGBA':
                        background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                        background.paste(img_pil, mask=img_pil.split()[3])
                        img_pil = background
                    elif img_pil.mode == 'LA':
                        background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                        background.paste(img_pil.convert('RGB'), mask=img_pil.split()[1])
                        img_pil = background
                    else:
                        img_pil = img_pil.convert('RGB')
                
                return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"Could not decode image from URL: {str(e)}")
        
        # Load from file
        # Try OpenCV first
        img = cv2.imread(source_str)
        if img is not None:
            return img
        
        # Fallback to PIL for formats OpenCV might not handle
        try:
            from PIL import Image as PILImage
            
            img_pil = PILImage.open(source_str)
            img_pil.load()
            
            if img_pil.mode not in ('RGB', 'L'):
                if img_pil.mode == 'P':
                    img_pil = img_pil.convert('RGB')
                elif img_pil.mode == 'RGBA':
                    background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                    background.paste(img_pil, mask=img_pil.split()[3])
                    img_pil = background
                elif img_pil.mode == 'LA':
                    background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                    background.paste(img_pil.convert('RGB'), mask=img_pil.split()[1])
                    img_pil = background
                else:
                    img_pil = img_pil.convert('RGB')
            
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Could not load image from {source_str}: {str(e)}")
    
    if isinstance(source, Image.Image):
        # Convert PIL Image to OpenCV format
        img_pil = source
        
        # Handle different PIL modes
        if img_pil.mode not in ('RGB', 'L'):
            if img_pil.mode == 'P':
                img_pil = img_pil.convert('RGB')
            elif img_pil.mode == 'RGBA':
                background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                background.paste(img_pil, mask=img_pil.split()[3])
                img_pil = background
            elif img_pil.mode == 'LA':
                background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                background.paste(img_pil.convert('RGB'), mask=img_pil.split()[1])
                img_pil = background
            else:
                img_pil = img_pil.convert('RGB')
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    raise ValueError(f"Unsupported image source type: {type(source)}")


def save_image(
    image: np.ndarray,
    path: str,
    quality: int = 95
) -> str:
    """
    Save image to file
    
    Args:
        image: numpy array (BGR format)
        path: Output file path
        quality: JPEG quality (1-100)
        
    Returns:
        Path to saved file
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine encoding format based on extension
    ext = path_obj.suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext in ['.png']:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    else:
        encode_params = []
    
    success = cv2.imwrite(str(path_obj), image, encode_params)
    if not success:
        raise IOError(f"Failed to save image to {path}")
    
    return str(path_obj)


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_thickness: int = 2,
    font_size: float = 0.5
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries with bbox, class_name, confidence
        show_labels: Whether to show class labels
        show_confidence: Whether to show confidence scores
        line_thickness: Thickness of bounding box lines
        font_size: Size of text font
        
    Returns:
        Image with drawn detections
    """
    img = image.copy()
    
    for i, detection in enumerate(detections):
        # Get bounding box coordinates
        bbox = detection.get('bbox')
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        else:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        class_name = detection.get('class_name', 'unknown')
        confidence = detection.get('confidence', 0)
        class_id = detection.get('class_id', 0)
        
        # Get color for this class
        color = COLORS[class_id % len(COLORS)]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label background and text
        if show_labels:
            label = class_name
            if show_confidence:
                label += f' {confidence:.2f}'
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1
            )
            
            # Draw label background
            cv2.rectangle(
                img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1  # Filled rectangle
            )
            
            # Draw text
            cv2.putText(
                img,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (0, 0, 0),  # Black text
                1
            )
    
    return img


def draw_detections_glow(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    glow_intensity: int = 10
) -> np.ndarray:
    """
    Draw bounding boxes with neon glow effect for futuristic UI
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dictionaries
        glow_intensity: Intensity of the glow effect
        
    Returns:
        Image with glow effect bounding boxes
    """
    img = image.copy()
    
    # Create overlay for glow effect
    overlay = img.copy()
    
    for detection in detections:
        bbox = detection.get('bbox')
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        else:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        class_id = detection.get('class_id', 0)
        color = COLORS[class_id % len(COLORS)]
        
        # Draw multiple layers for glow effect
        for thickness in range(glow_intensity, 0, -2):
            alpha = 1.0 - (thickness / glow_intensity) * 0.7
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        
        # Blend overlay with original
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Draw final sharp border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add class label with glow
        class_name = detection.get('class_name', 'unknown')
        confidence = detection.get('confidence', 0)
        label = f'{class_name} {confidence:.2f}'
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Glow background
        cv2.rectangle(
            overlay,
            (x1, y1 - text_height - baseline - 8),
            (x1 + text_width + 8, y1),
            color,
            -1
        )
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # Text
        cv2.putText(
            img,
            label,
            (x1 + 3, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img


def resize_image(
    image: np.ndarray,
    max_size: int = None,
    width: int = None,
    height: int = None,
    maintain_aspect: bool = True
) -> np.ndarray:
    """
    Resize image with various options
    
    Args:
        image: Input image
        max_size: Maximum dimension (width or height)
        width: Target width
        height: Target height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if max_size and (h > max_size or w > max_size):
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * w * max_size / w)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if width and height:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    if width:
        ratio = width / w
        new_h = int(h * ratio)
        return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)
    
    if height:
        ratio = height / h
        new_w = int(w * ratio)
        return cv2.resize(image, (new_w, height), interpolation=cv2.INTER_AREA)
    
    return image


def validate_image(
    image_data: bytes,
    max_size: int = -1
) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Validate image file
    
    Args:
        image_data: Image bytes
        max_size: Maximum file size in bytes (-1 for unlimited)
        
    Returns:
        (is_valid, error_message, image_array)
    """
    # Check file size (skip if max_size is -1 or None)
    if max_size is not None and max_size > 0 and len(image_data) > max_size:
        return False, f"File size exceeds maximum ({max_size} bytes)", None
    
    # Check for empty data
    if not image_data or len(image_data) == 0:
        return False, "Empty image file", None
    
    # Try to decode using OpenCV first
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            # Check dimensions
            h, w = img.shape[:2]
            if w < 10 or h < 10:
                return False, "Image dimensions too small (minimum 10x10 pixels)", None
            
            if w > 100000 or h > 100000:
                return False, "Image dimensions too large (maximum 100000x100000 pixels)", None
            
            return True, "Valid image", img
    except Exception as e:
        logger.warning(f"OpenCV decode failed: {e}")
    
    # Fallback: Try PIL for formats OpenCV might not handle well
    try:
        from PIL import Image as PILImage
        import io
        
        img_pil = PILImage.open(io.BytesIO(image_data))
        img_pil.load()  # Force load to detect corrupted images
        
        # Convert to RGB if necessary (handle RGBA, palette modes, etc.)
        if img_pil.mode not in ('RGB', 'L'):
            # Handle P (palette) mode and other modes
            if img_pil.mode == 'P':
                img_pil = img_pil.convert('RGB')
            elif img_pil.mode == 'RGBA':
                # Convert RGBA to RGB with white background
                background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                background.paste(img_pil, mask=img_pil.split()[3])  # 3 is the alpha channel
                img_pil = background
            elif img_pil.mode == 'LA':
                # Grayscale with alpha
                background = PILImage.new('RGB', img_pil.size, (255, 255, 255))
                background.paste(img_pil.convert('RGB'), mask=img_pil.split()[1])
                img_pil = background
            else:
                img_pil = img_pil.convert('RGB')
        
        # Convert PIL to OpenCV format
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Check dimensions
        h, w = img.shape[:2]
        if w < 10 or h < 10:
            return False, "Image dimensions too small (minimum 10x10 pixels)", None
        
        if w > 100000 or h > 100000:
            return False, "Image dimensions too large (maximum 100000x100000 pixels)", None
        
        return True, "Valid image", img
        
    except Exception as e:
        logger.error(f"PIL decode failed: {e}")
        return False, f"Invalid or corrupted image file: {str(e)}", None


def encode_image_base64(
    image: np.ndarray,
    format: str = 'JPEG',
    quality: int = 90
) -> str:
    """
    Encode image to base64 string
    
    Args:
        image: numpy array (BGR format)
        format: Output format (JPEG, PNG, etc.)
        quality: JPEG quality (1-100)
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    # Determine encoding parameters
    if format.upper() == 'JPEG':
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = 'jpg'
    elif format.upper() == 'PNG':
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        ext = 'png'
    else:
        encode_params = []
        ext = format.lower()
    
    # Encode image
    success, buffer = cv2.imencode(f'.{ext}', image, encode_params)
    if not success:
        raise ValueError(f"Could not encode image to {format}")
    
    # Convert to base64
    b64_string = base64.b64encode(buffer).decode('utf-8')
    
    # Return with data URI prefix
    mime_type = f'image/{ext}'
    return f'data:{mime_type};base64,{b64_string}'


def generate_detection_filename(original_name: str = None) -> str:
    """Generate a unique filename for detection output"""
    timestamp = uuid.uuid4().hex[:8]
    if original_name:
        name_parts = Path(original_name).stem
        return f"detection_{name_parts}_{timestamp}.jpg"
    return f"detection_{timestamp}.jpg"


def get_image_stats(image: np.ndarray) -> Dict[str, Any]:
    """Get basic statistics about an image"""
    h, w = image.shape[:2]
    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 2),
        "total_pixels": w * h,
        "mean_brightness": float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))
    }