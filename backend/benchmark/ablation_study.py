"""
Ablation study for LenScope adaptive routing features.

Evaluates routing variants on a COCO val2017 subset and writes Table 3 assets:
- benchmark/results/ablation_results.csv
- benchmark/results/ablation_chart.png
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.detection import DetectionService
from backend.app.services.meta_dataset_collector import MetaDatasetCollector
from backend.app.utils.image import load_image
from backend.benchmark.run_experiments import (
    EXPERIMENT_MODELS,
    ImageSample,
    compute_map_at_threshold,
    load_coco_samples,
)


DATA_PATH = PROJECT_ROOT / "backend" / "data" / "meta_training_data.jsonl"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
ABLATION_CSV = RESULTS_DIR / "ablation_results.csv"
ABLATION_CHART = RESULTS_DIR / "ablation_chart.png"
VARIANT_MODEL_DIR = PROJECT_ROOT / "backend" / "weights" / "ablation"

FEATURE_NAMES = [
    "mean_brightness",
    "contrast",
    "blur_score",
    "estimated_object_count",
    "aspect_ratio",
    "resolution_small",
    "resolution_medium",
    "resolution_large",
    "scene_indoor",
    "scene_outdoor",
    "scene_crowd",
    "scene_vehicle",
    "scene_night",
]
RESOLUTION_BUCKETS = ["small", "medium", "large"]
SCENE_TYPES = ["indoor", "outdoor", "crowd", "vehicle", "night"]


@dataclass(frozen=True)
class AblationVariant:
    key: str
    label: str
    feature_names: List[str]
    baseline_model: Optional[str] = None


VARIANTS = [
    AblationVariant("v1", "V1: YOLOv8n always", [], baseline_model="yolov8n"),
    AblationVariant("v2", "V2: Brightness + contrast", ["mean_brightness", "contrast"]),
    AblationVariant(
        "v3",
        "V3: + Blur + object count",
        ["mean_brightness", "contrast", "blur_score", "estimated_object_count"],
    ),
    AblationVariant(
        "v4",
        "V4: All except CLIP scene",
        [
            "mean_brightness",
            "contrast",
            "blur_score",
            "estimated_object_count",
            "aspect_ratio",
            "resolution_small",
            "resolution_medium",
            "resolution_large",
        ],
    ),
    AblationVariant("v5", "V5: Full system", FEATURE_NAMES),
]


def load_meta_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_map_score(row: Dict[str, Any]) -> Optional[float]:
    for key in ("map_score", "mAP", "map", "mAP_score"):
        if row.get(key) is not None:
            return float(row[key])
    return None


def best_rows_by_scene(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        features = row.get("scene_features")
        model_id = row.get("model_used") or row.get("model_id")
        score = get_map_score(row)
        if not features or not model_id or score is None:
            continue
        key = json.dumps(features, sort_keys=True, separators=(",", ":"))
        if key not in best or score > get_map_score(best[key]):
            best[key] = row
    return list(best.values())


def full_feature_vector(scene_features: Dict[str, Any]) -> Dict[str, float]:
    bucket = scene_features.get("resolution_bucket")
    scene_type = scene_features.get("scene_type")
    values = {
        "mean_brightness": float(scene_features.get("mean_brightness", 0.0) or 0.0),
        "contrast": float(scene_features.get("contrast", 0.0) or 0.0),
        "blur_score": float(scene_features.get("blur_score", 0.0) or 0.0),
        "estimated_object_count": float(scene_features.get("estimated_object_count", 0.0) or 0.0),
        "aspect_ratio": float(scene_features.get("aspect_ratio", 0.0) or 0.0),
    }
    values.update({f"resolution_{name}": 1.0 if bucket == name else 0.0 for name in RESOLUTION_BUCKETS})
    values.update({f"scene_{name}": 1.0 if scene_type == name else 0.0 for name in SCENE_TYPES})
    return values


def vectorize(scene_features: Dict[str, Any], feature_names: List[str]) -> List[float]:
    values = full_feature_vector(scene_features)
    return [values[name] for name in feature_names]


def train_or_load_router(
    variant: AblationVariant,
    rows: List[Dict[str, Any]],
    force_retrain: bool,
) -> Optional[Dict[str, Any]]:
    if variant.baseline_model:
        return None

    model_path = VARIANT_MODEL_DIR / f"{variant.key}_router.pkl"
    if model_path.exists() and not force_retrain:
        with model_path.open("rb") as file:
            return pickle.load(file)

    labeled_rows = best_rows_by_scene(rows)
    if len(labeled_rows) < 2:
        print(f"{variant.key}: insufficient labeled meta rows; falling back to yolov8n")
        return None

    labels = np.array([row.get("model_used") or row.get("model_id") for row in labeled_rows])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    if len(label_encoder.classes_) < 2:
        print(f"{variant.key}: only one label class; falling back to yolov8n")
        return None

    x = np.array(
        [vectorize(row["scene_features"], variant.feature_names) for row in labeled_rows],
        dtype=np.float32,
    )
    model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)

    bundle = {
        "variant": variant.key,
        "feature_names": variant.feature_names,
        "label_encoder": label_encoder,
        "model": model,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(bundle, file)
    return bundle


def predict_model(
    variant: AblationVariant,
    bundle: Optional[Dict[str, Any]],
    scene_features: Dict[str, Any],
) -> str:
    if variant.baseline_model or bundle is None:
        return variant.baseline_model or "yolov8n"

    x = np.array([vectorize(scene_features, bundle["feature_names"])], dtype=np.float32)
    prediction = bundle["model"].predict(x)[0]
    return str(bundle["label_encoder"].inverse_transform([prediction])[0])


def evaluate_variant(
    variant: AblationVariant,
    bundle: Optional[Dict[str, Any]],
    samples: List[ImageSample],
    service: DetectionService,
    collector: MetaDatasetCollector,
    confidence: float,
) -> Dict[str, Any]:
    predictions_by_image: Dict[str, List[Dict[str, Any]]] = {}
    routing_overheads = []

    for sample in tqdm(samples, desc=variant.key, leave=False):
        image = load_image(sample.image_path)
        start = time.perf_counter()
        if variant.baseline_model:
            model_id = variant.baseline_model
        else:
            scene_features = collector._extract_scene_features(image)
            model_id = predict_model(variant, bundle, scene_features)
        routing_overheads.append((time.perf_counter() - start) * 1000)

        if model_id not in EXPERIMENT_MODELS:
            model_id = "yolov8n"
        engine = service.get_engine(model_id)
        detections, _ = engine.detect(
            image,
            confidence_threshold=confidence,
            iou_threshold=0.45,
            max_detections=100,
        )
        predictions_by_image[sample.image_id] = detections

    return {
        "variant": variant.key,
        "description": variant.label,
        "num_features": len(variant.feature_names),
        "mAP@0.5": compute_map_at_threshold(samples, predictions_by_image, 0.5),
        "mean_latency_overhead_ms": float(np.mean(routing_overheads)) if routing_overheads else 0.0,
    }


def save_chart(results: pd.DataFrame) -> None:
    labels = results["variant"].tolist()
    map_values = results["mAP@0.5"].astype(float).tolist()
    overhead_values = results["mean_latency_overhead_ms"].astype(float).tolist()

    x = np.arange(len(labels))
    width = 0.38
    fig, axis_left = plt.subplots(figsize=(9, 5))
    axis_right = axis_left.twinx()

    bars_map = axis_left.bar(x - width / 2, map_values, width, label="mAP@0.5", color="#2563eb")
    bars_latency = axis_right.bar(x + width / 2, overhead_values, width, label="Routing overhead (ms)", color="#f97316")

    axis_left.set_ylabel("mAP@0.5")
    axis_right.set_ylabel("Mean routing overhead (ms)")
    axis_left.set_xlabel("Ablation variant")
    axis_left.set_xticks(x)
    axis_left.set_xticklabels(labels)
    axis_left.set_ylim(0, max(max(map_values) * 1.2, 0.05))
    axis_left.grid(axis="y", alpha=0.25)
    axis_left.set_title("Adaptive Routing Feature Ablation")

    handles = [bars_map, bars_latency]
    axis_left.legend(handles, [handle.get_label() for handle in handles], loc="upper left")
    fig.tight_layout()
    fig.savefig(ABLATION_CHART, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_coco_samples(Path(args.coco_val2017), quick=False)[: args.limit]
    if not samples:
        raise ValueError("No COCO val2017 samples found")

    rows = load_meta_rows(Path(args.meta_data))
    service = DetectionService()
    collector = MetaDatasetCollector()
    results = []

    for variant in tqdm(VARIANTS, desc="Ablation variants"):
        bundle = train_or_load_router(variant, rows, force_retrain=args.force_retrain)
        results.append(evaluate_variant(variant, bundle, samples, service, collector, args.confidence))

    results_df = pd.DataFrame(results)
    results_df.to_csv(ABLATION_CSV, index=False)
    save_chart(results_df)
    print(f"Saved ablation results to {ABLATION_CSV}")
    print(f"Saved ablation chart to {ABLATION_CHART}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive routing ablation study")
    parser.add_argument("--coco-val2017", required=True, help="Path to COCO root containing val2017 and annotations")
    parser.add_argument("--meta-data", default=str(DATA_PATH), help="Path to meta_training_data.jsonl")
    parser.add_argument("--limit", type=int, default=500, help="Number of COCO val2017 images to evaluate")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain ablation routers even if cached")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
