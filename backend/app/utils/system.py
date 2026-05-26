"""
System utility functions for hardware detection and monitoring
"""

import torch
import platform
import psutil
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def check_cuda_available() -> bool:
    """Check if CUDA is available"""
    return torch.cuda.is_available()


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """Get GPU information if available"""
    if not torch.cuda.is_available():
        return None
    
    try:
        gpu_info = {
            "count": torch.cuda.device_count(),
            "devices": []
        }
        
        for i in range(torch.cuda.device_count()):
            device_info = {
                "name": torch.cuda.get_device_name(i),
                "compute_capability": torch.cuda.get_device_capability(i),
                "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3)
            }
            gpu_info["devices"].append(device_info)
        
        return gpu_info
    except Exception as e:
        logger.warning(f"Could not get GPU info: {e}")
        return None


def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information"""
    try:
        return {
            "processor": platform.processor() or "Unknown",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
    except Exception as e:
        logger.warning(f"Could not get CPU info: {e}")
        return {
            "processor": "Unknown",
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True)
        }


def get_memory_info() -> Dict[str, Any]:
    """Get system memory information"""
    try:
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent_used": mem.percent
        }
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        return {}


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_info": get_gpu_info(),
        "cpu_info": get_cpu_info(),
        "memory_info": get_memory_info()
    }


def get_optimal_device() -> str:
    """Get the optimal compute device based on availability"""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def can_use_fp16() -> bool:
    """Check if FP16 (half precision) is supported"""
    if not torch.cuda.is_available():
        return False
    
    try:
        # FP16 is well supported on compute capability >= 5.3
        major, minor = torch.cuda.get_device_capability(0)
        return major >= 5 or (major == 5 and minor >= 3)
    except:
        return False


def can_use_int8() -> bool:
    """Check if INT8 quantization is supported"""
    if not torch.cuda.is_available():
        return False
    
    try:
        # INT8 is well supported on compute capability >= 6.1
        major, minor = torch.cuda.get_device_capability(0)
        return major >= 7 or (major == 6 and minor >= 1)
    except:
        return False


def get_recommended_batch_size() -> int:
    """Get recommended batch size based on available memory"""
    if not torch.cuda.is_available():
        return 1
    
    try:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory_gb >= 24:
            return 32
        elif gpu_memory_gb >= 16:
            return 16
        elif gpu_memory_gb >= 8:
            return 8
        elif gpu_memory_gb >= 4:
            return 4
        else:
            return 2
    except:
        return 1


def monitor_gpu_usage() -> Dict[str, float]:
    """Monitor current GPU usage"""
    if not torch.cuda.is_available():
        return {}
    
    try:
        return {
            "memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "memory_percent": (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
        }
    except:
        return {}


def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()