r"""
Train or fine-tune local Ultralytics YOLO models on a real-world dataset.

Examples:
    python scripts/train_yolo_models.py --data coco.yaml --models yolov8n yolov8s --epochs 50
    python scripts/train_yolo_models.py --data C:\datasets\mydata\data.yaml --models all --epochs 100

The dataset must be an Ultralytics-compatible YAML file. For custom objects,
label images in YOLO format and point --data at that dataset YAML.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import AVAILABLE_MODELS, settings


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local YOLO models.")
    parser.add_argument(
        "--data",
        required=True,
        help="Ultralytics dataset YAML path or name, for example coco.yaml or C:\\datasets\\objects\\data.yaml.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n"],
        help="Model ids to train, or 'all' for local YOLOv8 weights found in weights/.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="auto", help="Batch size integer or 'auto'.")
    parser.add_argument("--device", default=None, help="Training device, for example cpu, 0, or 0,1.")
    parser.add_argument("--project", default=str(settings.project_root / "runs" / "train"))
    parser.add_argument("--name-prefix", default="lenscope")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def local_weight_path(model_id: str) -> Path | None:
    config = AVAILABLE_MODELS.get(model_id)
    if not config:
        return None

    weights = Path(config["weights"])
    candidates = [
        weights if weights.is_absolute() else None,
        settings.weights_path / weights.name,
        settings.project_root / weights.name,
        settings.project_root / "backend" / "weights" / weights.name,
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    return None


def selectable_models(requested: Iterable[str]) -> list[str]:
    requested = list(requested)
    if requested == ["all"]:
        requested = [
            model_id
            for model_id, config in AVAILABLE_MODELS.items()
            if config.get("runnable", False)
            and config.get("backend") == "ultralytics"
            and model_id.startswith("yolov8")
            and local_weight_path(model_id)
        ]

    unknown = [model_id for model_id in requested if model_id not in AVAILABLE_MODELS]
    if unknown:
        raise ValueError(f"Unknown model ids: {', '.join(unknown)}")

    unsupported = [
        model_id
        for model_id in requested
        if not AVAILABLE_MODELS[model_id].get("runnable", False)
        or AVAILABLE_MODELS[model_id].get("backend") != "ultralytics"
    ]
    if unsupported:
        raise ValueError(
            "Only runnable Ultralytics models can be trained by this script. "
            f"Unsupported: {', '.join(unsupported)}"
        )

    missing = [model_id for model_id in requested if not local_weight_path(model_id)]
    if missing:
        raise FileNotFoundError(
            "Missing local weight files. Put them in weights/ first: "
            f"{', '.join(missing)}"
        )

    return requested


def train_model(model_id: str, args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    weights_path = local_weight_path(model_id)
    if not weights_path:
        raise FileNotFoundError(f"No local weights found for {model_id}")

    run_name = f"{args.name_prefix}-{model_id}"
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": run_name,
        "resume": args.resume,
    }

    if args.device:
        train_args["device"] = args.device

    logging.info("Training %s with %s", model_id, json.dumps(train_args, indent=2))

    if args.dry_run:
        return {"model_id": model_id, "weights": str(weights_path), "train_args": train_args}

    model = YOLO(str(weights_path))
    result = model.train(**train_args)
    best_path = Path(args.project) / run_name / "weights" / "best.pt"
    return {
        "model_id": model_id,
        "source_weights": str(weights_path),
        "best_weights": str(best_path),
        "result": str(result),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args()
    models = selectable_models(args.models)

    logging.info("Selected models: %s", ", ".join(models))

    results = []
    for model_id in models:
        results.append(train_model(model_id, args))

    print(json.dumps({"trained": results}, indent=2))


if __name__ == "__main__":
    main()
