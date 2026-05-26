# LenScope AI Project Documentation

This document captures how the current LenScope codebase is organized, how the services communicate, and how to operate the dashboard during development.

## System Overview

LenScope AI has two primary applications:

- `frontend/`: a Next.js dashboard that provides the browser experience for uploads, webcam detection, model selection, analytics, admin views, and advanced AI panels.
- `backend/`: a FastAPI service that exposes detection, model management, analytics, admin, authentication, and AI feature APIs.

The frontend calls the backend through `NEXT_PUBLIC_API_URL`, which defaults to `http://localhost:8000`. The backend enables CORS for `http://localhost:3000` and `http://127.0.0.1:3000`.

The repository also includes a standalone root `index.html` dashboard for quick local demos. It calls the same backend APIs directly from browser JavaScript.

## Runtime Flow

1. A user opens the dashboard at `http://localhost:3000`.
2. The dashboard sends API requests to the FastAPI backend at `http://localhost:8000`.
3. Detection requests upload image data to `/api/detection/detect/image`.
4. The backend validates the image, loads or reuses the selected model, runs inference, and returns detection metadata.
5. Generated output files are written under the configured upload directory and are served from `/uploads`.
6. AI feature endpoints maintain in-memory scene, relationship, prediction, and explanation state for dashboard panels.

## Backend

Entry point: `backend/main.py`

Key responsibilities:

- Create the FastAPI app.
- Initialize runtime directories and the database during lifespan startup.
- Register API routers.
- Configure CORS.
- Serve generated upload files from `/uploads`.
- Expose `/health`, `/api/docs`, `/api/redoc`, and `/api/openapi.json`.

Important backend modules:

- `backend/app/config.py` - environment-backed settings, available model registry, COCO classes.
- `backend/app/database.py` - database initialization and session handling.
- `backend/app/routes/` - HTTP API route groups.
- `backend/app/services/detection.py` - object detection service.
- `backend/app/services/model_manager.py` - model loading, device selection, model metadata.
- `backend/app/services/adaptive_ai.py` - adaptive model recommendation/switching.
- `backend/app/services/scene_memory.py` - scene timeline and object tracking memory.
- `backend/app/services/object_relationships.py` - object relationship graph.
- `backend/app/services/prediction_engine.py` - future scene and trajectory prediction.
- `backend/app/services/explanation_engine.py` - explainability summaries.

## Frontend

Entry point: `frontend/src/app/page.tsx`

Important frontend pages:

- `/` - landing dashboard.
- `/detect` - image upload detection.
- `/webcam` - webcam detection workflow with live boxes, object counts, and up/down movement counters.
- `/analytics` - analytics dashboard.
- `/models` - model inventory, model status, and comparison metadata.
- `/admin` - administrative controls and system views.
- `/ai-features` - adaptive AI, scene memory, graph, predictions, and explanations.

Important frontend configuration:

- `frontend/next.config.js` includes image host configuration and an `/api/:path*` rewrite to the local backend.
- `frontend/.env.local` sets `NEXT_PUBLIC_API_URL=http://localhost:8000`.
- `frontend/src/app/globals.css` contains the global visual system.

## API Map

### Health and Docs

- `GET /health` - simple backend health check.
- `GET /api/docs` - Swagger UI.
- `GET /api/redoc` - ReDoc UI.

### Detection

- `GET /api/detection/health` - detection service health and system status.
- `GET /api/detection/models` - detection model list and system recommendation.
- `GET /api/detection/classes` - COCO class list.
- `POST /api/detection/detect/image` - run object detection on an uploaded image.
- `POST /api/detection/detect/compare` - compare multiple models on one image.
- `GET /api/detection/recommend` - recommend a model for a use case.
- `GET /api/detection/system-info` - diagnostics and loaded model state.

### Models

- `GET /api/models/` - all available models grouped by type.
- `GET /api/models/recommend` - model recommendation.
- `GET /api/models/system-capabilities` - system capability summary.
- `GET /api/models/loaded` - currently loaded models.
- `POST /api/models/unload/{model_id}` - unload one model.
- `POST /api/models/unload-all` - unload all models.
- `GET /api/models/memory-usage` - memory usage summary.

### AI Features

- `GET /api/ai/adaptive/status`
- `GET /api/ai/adaptive/switch-history`
- `GET /api/ai/adaptive/performance-stats`
- `POST /api/ai/adaptive/reset`
- `GET /api/ai/scene/timeline`
- `GET /api/ai/scene/events`
- `GET /api/ai/scene/history`
- `GET /api/ai/scene/statistics`
- `GET /api/ai/scene/tracks`
- `GET /api/ai/graph/overview`
- `GET /api/ai/graph/full`
- `GET /api/ai/predictions/scene-evolution`
- `GET /api/ai/predictions/next-frame`
- `GET /api/ai/explanations/history`
- `GET /api/ai/analysis/full`
- `GET /api/ai/recommend/model`
- `GET /api/ai/hardware/resources`
- `GET /api/ai/models/list`
- `GET /api/ai/recommend/scenarios`

Use `/api/docs` for request/response schemas and the complete parameter list.

## Local Development Workflow

Run backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Run frontend:

```powershell
cd frontend
npm install
npm run dev
```

Verify:

```powershell
Invoke-WebRequest http://localhost:8000/health
Invoke-WebRequest http://localhost:3000
```

## Model Weights

The default model is `yolov8n.pt`. The active runtime catalog returned by `/api/detection/models` includes 17 entries:

- YOLOv8: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- Open vocabulary: `yolov8s-world`
- SSD/SSDLite: `mobilenet-ssd`, `ssd300-vgg16`, `ssdlite-mobilenetv3`
- Faster R-CNN: `faster-rcnn`, `faster-rcnn-r50`, `faster-rcnn-r101`
- RetinaNet: `retinanet`, `retinanet-r50-fpn`
- EfficientDet: `efficientdet-d0`, `efficientdet-d1`, `efficientdet-d2`

The non-YOLO families are runtime-safe comparison entries using the current Ultralytics inference path with local YOLO weights as proxies. This keeps model selection and comparison stable while native SSD/Faster R-CNN/RetinaNet/EfficientDet loaders are not yet wired into `ModelManager`.

Keep model files in the configured `WEIGHTS_DIR`. The default setting is `./weights` from the project root as resolved by `backend/app/config.py`.

After changing `backend/app/config.py`, restart the backend process. The running FastAPI server keeps the model registry in memory, so API responses will not reflect catalog changes until restart.

## Standalone Dashboard

Run the root `index.html` dashboard through a localhost web server:

```powershell
cd C:\LenScope
python -m http.server 5500
```

Then open `http://localhost:5500/index.html`.

Notes:

- Browser camera access requires a secure context. Use `localhost`, `127.0.0.1`, or HTTPS.
- Do not open `index.html` directly as `file://` for webcam workflows.
- The standalone Models segment tries backend model APIs first and then renders a built-in fallback model list if the backend is unreachable.
- The standalone webcam segment sends frames to `/api/detection/detect/image`, draws returned boxes, tracks object count, and counts vertical up/down movement based on object center movement between frames.

## Data and Generated Files

- `lenscope.db` - SQLite database file.
- `uploads/` - uploaded/generated output files.
- `weights/` - model weights.
- `logs/` - application logs.
- `frontend/.next/` - Next.js build/dev output.
- `frontend/node_modules/` - installed frontend dependencies.

These runtime/generated directories should usually be kept out of source control in a normal Git setup.

## Docker Notes

Run both services:

```powershell
docker compose up --build
```

The current Compose configuration exposes:

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

The backend Dockerfile installs Python dependencies and starts Uvicorn. The frontend Dockerfile builds the Next.js app and starts it as a production server.

## Troubleshooting

Backend import or dependency errors:

- Confirm the virtual environment is active.
- Reinstall dependencies with `python -m pip install -r backend/requirements.txt`.
- Check that your Python version is compatible with the installed PyTorch package.

Frontend cannot reach backend:

- Confirm the backend is running at `http://localhost:8000`.
- Check `frontend/.env.local`.
- Confirm CORS origins in `backend/.env` include the frontend URL.

Models page stays on loading or shows old models:

- Confirm `http://localhost:8000/api/detection/models` returns JSON in the browser.
- Restart the backend after model catalog edits.
- Hard refresh the dashboard with `Ctrl + F5`.
- In the standalone dashboard, check whether the model count label says `backend` or `fallback`.

Detection returns no results:

- Lower the confidence threshold.
- Confirm the selected model exists in the model registry.
- Confirm the corresponding weight file is available.
- Check backend logs for model loading errors.

Webcam does not start:

- Open the dashboard from `http://localhost...`, not `file://` or a LAN IP.
- Allow browser camera permission.
- Close other applications using the camera.
- Use `YOLOv8m` or `YOLOv8s-World` with confidence around `25-35%` when non-person objects are missed.

CUDA/GPU is not used:

- Confirm the installed PyTorch build supports CUDA.
- Check `/api/detection/system-info`.
- Set `DEVICE=cuda` only when CUDA is available; otherwise keep `DEVICE=auto`.

Docker frontend fails to start:

- Confirm the production build output matches the Docker start command.
- If using standard Next.js production mode, `npm run start` is often simpler than a standalone `server.js` command unless standalone output is configured.

## Maintenance Checklist

- Keep `README.md` focused on setup and quick orientation.
- Keep this document updated when adding routes, services, environment variables, or runtime directories.
- Prefer API schema updates in FastAPI models and validate changes through `/api/docs`.
- Verify both `http://localhost:8000/health` and `http://localhost:3000` after dependency or configuration changes.
