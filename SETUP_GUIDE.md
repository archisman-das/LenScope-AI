# LenScope AI Setup Guide

This guide covers the local backend, Next.js frontend, and optional standalone `index.html` dashboard.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- Model weights such as `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, and `yolov8x.pt` in the project or `weights/` directory
- Optional: CUDA-capable NVIDIA GPU with a CUDA-enabled PyTorch install

## Backend

```powershell
cd C:\LenScope\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Backend URL:

```text
http://localhost:8000
```

Useful checks:

```powershell
Invoke-WebRequest http://localhost:8000/health
Invoke-WebRequest http://localhost:8000/api/detection/models
```

Swagger docs:

```text
http://localhost:8000/api/docs
```

## Frontend

```powershell
cd C:\LenScope\frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:3000
```

The frontend uses `NEXT_PUBLIC_API_URL` from `frontend/.env.local`. For local development, use:

```text
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Standalone Dashboard

The root `index.html` can run without the Next.js dev server:

```powershell
cd C:\LenScope
python -m http.server 5500
```

Open:

```text
http://localhost:5500/index.html
```

Do not open `index.html` through `file://` for webcam testing. Browsers require camera access to run from `localhost`, `127.0.0.1`, or HTTPS.

## Models

The active backend catalog includes:

- YOLOv8: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- Open vocabulary: `yolov8s-world`
- SSD/SSDLite: `mobilenet-ssd`, `ssd300-vgg16`, `ssdlite-mobilenetv3`
- Faster R-CNN: `faster-rcnn`, `faster-rcnn-r50`, `faster-rcnn-r101`
- RetinaNet: `retinanet`, `retinanet-r50-fpn`
- EfficientDet: `efficientdet-d0`, `efficientdet-d1`, `efficientdet-d2`

SSD, Faster R-CNN, RetinaNet, and EfficientDet are currently runtime-safe comparison entries that use the existing Ultralytics inference path with available YOLO weights as proxies.

After editing `backend/app/config.py`, restart the backend before checking the Models dashboard:

```powershell
netstat -ano | Select-String ':8000'
Stop-Process -Id <PID> -Force
cd C:\LenScope\backend
python main.py
```

## Webcam Notes

- Use `http://localhost...` and allow camera permission.
- Close other camera apps if the camera cannot start.
- Use `YOLOv8m` or `YOLOv8s-World` when non-person objects are missed.
- Lower confidence to around `25-35%` for smaller objects.
- The dashboard counts detected objects plus vertical up/down movement between frames.

## Quick Troubleshooting

Backend not reachable:

```powershell
Invoke-WebRequest http://localhost:8000/health
```

Models page still shows old models:

- Restart backend.
- Refresh the browser with `Ctrl + F5`.
- Check `http://localhost:8000/api/detection/models`.

Frontend build check:

```powershell
cd C:\LenScope\frontend
npm run build
```

Standalone HTML syntax check:

```powershell
cd C:\LenScope
node -e "const fs=require('fs'); const html=fs.readFileSync('index.html','utf8'); const re=new RegExp('<script(?:\\\\s[^>]*)?>([\\\\s\\\\S]*?)<\\\\/script>','gi'); const scripts=[...html.matchAll(re)].map(m=>m[1]).filter(s=>!s.includes('tailwind.config')); for (const script of scripts) new Function(script); console.log('inline script syntax ok')"
```
