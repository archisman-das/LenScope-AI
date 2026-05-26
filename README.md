# LenScope AI

LenScope AI is a full-stack object detection dashboard for image uploads, webcam workflows, model comparison, analytics, and advanced AI scene analysis. The frontend is a Next.js app, and the backend is a FastAPI service that runs YOLOv8-family detection with CPU/GPU-aware model management.

## What It Includes

- Image-based object detection with configurable confidence and IoU thresholds.
- Live webcam detection workflows from the browser, including bounding boxes, object counts, and up/down movement counters.
- Multi-model comparison and model recommendation tools.
- A 17-entry model catalog covering YOLOv8, YOLO-World, SSD/SSDLite, Faster R-CNN, RetinaNet, and EfficientDet comparison entries.
- Analytics views for detections, model usage, and runtime performance.
- Advanced AI features for adaptive model switching, scene memory, object relationships, predictions, and explanation history.
- Standalone `index.html` dashboard for lightweight local demos alongside the Next.js frontend.
- Docker Compose support for running frontend and backend together.

## Tech Stack

- Frontend: Next.js 14, React 18, TypeScript, Tailwind CSS, Framer Motion, Chart.js.
- Backend: FastAPI, Uvicorn, SQLAlchemy async, SQLite, PyTorch, Ultralytics YOLO, OpenCV.
- Runtime assets: model weights in `weights/`, uploads and generated output in `uploads/`, logs in `logs/`.

## Project Structure

```text
LenScope/
  backend/              FastAPI application, routes, services, schemas, Dockerfile
  frontend/             Next.js dashboard application, Dockerfile
  docs/                 Project documentation
  index.html            Standalone browser dashboard
  uploads/              Runtime upload/output directory
  weights/              Runtime model weight directory
  logs/                 Runtime logs
  docker-compose.yml    Full-stack container setup
```

## Prerequisites

- Node.js 18 or newer.
- Python 3.10 or newer.
- A YOLO weight file such as `yolov8n.pt` in `weights/` or `backend/weights/`, depending on how you run the backend.
- Optional: NVIDIA GPU and CUDA-compatible PyTorch for accelerated inference.
- Optional: Docker and Docker Compose.

## Local Development

### 1. Start the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

The API runs at `http://localhost:8000`.

Useful backend URLs:

- Health check: `http://localhost:8000/health`
- Detection health: `http://localhost:8000/api/detection/health`
- Swagger docs: `http://localhost:8000/api/docs`

### 2. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

The dashboard runs at `http://localhost:3000`.

The frontend reads `NEXT_PUBLIC_API_URL` from `frontend/.env.local`; by default it points to `http://localhost:8000`.

### Optional: run the standalone dashboard

The root `index.html` can be served without the Next.js dev server:

```powershell
cd C:\LenScope
python -m http.server 5500
```

Open `http://localhost:5500/index.html`. Webcam access requires `localhost`, `127.0.0.1`, or HTTPS; opening the file through `file://` or a LAN IP can block the camera.

## Docker

```powershell
docker compose up --build
```

Services:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

The compose file maps backend port `8000` and frontend port `3000`, and passes `NEXT_PUBLIC_API_URL=http://localhost:8000` to the frontend.

## Main App Routes

Frontend pages:

- `/` - dashboard landing page
- `/detect` - image upload detection
- `/webcam` - live webcam workflow with movement counters
- `/analytics` - detection analytics
- `/models` - model list, recommendations, and system capability views
- `/admin` - administrative dashboard
- `/ai-features` - adaptive AI, scene memory, prediction, and explanation tools

Backend API groups:

- `/api/detection/*` - health, classes, image detection, model comparison, recommendations, system info
- `/api/models/*` - available models, loaded models, memory usage, unload controls
- `/api/analytics/*` - analytics data
- `/api/admin/*` - admin/system controls
- `/api/auth/*` - authentication endpoints
- `/api/ai/*` - adaptive AI, scene timeline, object graph, predictions, explanations, hardware resources

## Configuration

Backend configuration lives in `backend/.env` and `backend/app/config.py`.

Important values:

- `DATABASE_URL` - defaults to SQLite at `sqlite+aiosqlite:///./lenscope.db`.
- `DEFAULT_MODEL` - defaults to `yolov8n.pt`.
- `DEVICE` - `auto`, `cpu`, or `cuda`.
- `CORS_ORIGINS` - must include the frontend origin.
- `UPLOAD_DIR`, `WEIGHTS_DIR`, `LOG_FILE` - runtime paths.

The exposed runtime model catalog currently includes:

- YOLOv8: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- Open vocabulary: `yolov8s-world`
- SSD family: `mobilenet-ssd`, `ssd300-vgg16`, `ssdlite-mobilenetv3`
- Faster R-CNN family: `faster-rcnn`, `faster-rcnn-r50`, `faster-rcnn-r101`
- RetinaNet family: `retinanet`, `retinanet-r50-fpn`
- EfficientDet family: `efficientdet-d0`, `efficientdet-d1`, `efficientdet-d2`

Non-YOLO families are exposed as runtime-safe comparison entries through the existing Ultralytics inference path, using available local YOLO weights as proxies until native framework loaders are implemented.

Frontend configuration lives in `frontend/.env.local`.

Important values:

- `NEXT_PUBLIC_API_URL` - backend base URL used by frontend API calls.
- `NEXT_PUBLIC_ENABLE_WEBCAM`, `NEXT_PUBLIC_ENABLE_ANALYTICS`, `NEXT_PUBLIC_ENABLE_ADMIN` - optional feature flags.

## Documentation

See [docs/PROJECT.md](docs/PROJECT.md) for architecture notes, API map, development workflow, troubleshooting, and maintenance guidance.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
