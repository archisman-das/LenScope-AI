"""
LenScope AI - Object Detection Dashboard Backend
Main FastAPI application entry point
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import time

from app.config import settings
from app.database import init_db, close_db
from app.routes import auth_router, detection_router, models_router, analytics_router, admin_router, ai_features_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting LenScope AI Backend...")
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Pre-load default model
    try:
        from app.services.model_manager import ModelManager
        manager = ModelManager()
        logger.info(f"System device: {manager.device}")
        logger.info(f"CUDA available: {manager.cuda_available}")
        
        if manager.cuda_available:
            logger.info(f"GPU: {manager.gpu_name} ({manager.gpu_memory:.1f} GB)")
        
        # Load recommended model
        recommended = manager.recommend_model("realtime")
        logger.info(f"Recommended model for realtime: {recommended}")
        
    except Exception as e:
        logger.warning(f"Could not initialize model manager: {e}")
    
    logger.info("LenScope AI Backend started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LenScope AI Backend...")
    await close_db()
    logger.info("Backend shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="LenScope AI Dashboard",
    description="AI-powered Object Detection Dashboard with YOLOv8",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Configure max upload size (set to unlimited when MAX_UPLOAD_SIZE is -1)
if settings.MAX_UPLOAD_SIZE > 0:
    app.max_upload_size = settings.MAX_UPLOAD_SIZE
else:
    # Set to 10GB max for practical purposes (FastAPI/Starlette default is 1MB)
    app.max_upload_size = 10 * 1024 * 1024 * 1024  # 10GB

# CORS configuration - Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(500)
async def internal_server_error(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }
    )


@app.exception_handler(404)
async def not_found(request: Request, exc: Exception):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )


# Include routers
app.include_router(auth_router)
app.include_router(detection_router)
app.include_router(models_router)
app.include_router(analytics_router)
app.include_router(admin_router)
app.include_router(ai_features_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LenScope AI Dashboard",
        "version": "1.0.0",
        "description": "AI-powered Object Detection Dashboard",
        "endpoints": {
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "health": "/api/detection/health",
            "models": "/api/models",
            "detect": "/api/detection/detect/image",
            "compare": "/api/detection/detect/compare"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


# Static files (for uploads)
static_path = Path(settings.upload_path)
static_path.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(static_path)), name="uploads")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )