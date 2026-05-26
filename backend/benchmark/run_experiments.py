"""
Run LenScope model and adaptive-routing experiments.

Expected dataset layouts:
- COCO: <root>/val2017/*.jpg and <root>/annotations/instances_val2017.json
- VOC 2012: <root>/JPEGImages, <root>/Annotations, optional <root>/ImageSets/Main/val.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import COCO_CLASSES
from backend.app.services.adaptive_ai import get_adaptive_switcher
from backend.app.services.detection import DetectionService
from backend.app.utils.image import load_image


RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
RESULTS_CSV = RESULTS_DIR / "full_experiment_results.csv"
TABLE1_TEX = RESULTS_DIR / "table1_model_comparison.tex"
TABLE2_TEX = RESULTS_DIR / "table2_adaptive_routing.tex"

EXPERIMENT_MODELS = [
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "ssd300-vgg16",
    "faster-rcnn-r50",
    "retinanet-r50-fpn",
]

IOU_THRESHOLDS = [round(value, 2) for value in np.arange(0.5, 1.0, 0.05)]
VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
VOC_TO_COCO = {
    "aeroplane": "airplane",
    "diningtable": "dining table",
    "motorbike": "motorcycle",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "tvmonitor": "tv",
}


@dataclass
class ImageSample:
    dataset: str
    image_id: str
    image_path: Path
    annotations: List[Dict[str, Any]]


@dataclass
class ModelEval:
    metrics: Dict[str, float]
    per_image_ap: Dict[str, float]
    inference_ms_by_image: Dict[str, float]
    predictions_by_image: Dict[str, List[Dict[str, Any]]]


def load_coco_samples(root: Path, quick: bool) -> List[ImageSample]:
    annotation_path = root / "annotations" / "instances_val2017.json"
    image_dir = root / "val2017"
    if not annotation_path.exists() or not image_dir.exists():
        raise FileNotFoundError(f"COCO val2017 layout not found under {root}")

    with annotation_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    categories = {item["id"]: item["name"] for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}
    annotations_by_image: Dict[int, List[Dict[str, Any]]] = {image_id: [] for image_id in images}

    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, width, height = ann["bbox"]
        annotations_by_image.setdefault(ann["image_id"], []).append({
            "class_name": categories[ann["category_id"]],
            "bbox": [x, y, x + width, y + height],
        })

    samples = []
    for image_id, image_info in images.items():
        image_path = image_dir / image_info["file_name"]
        if image_path.exists():
            samples.append(ImageSample("coco_val2017", str(image_id), image_path, annotations_by_image.get(image_id, [])))

    return samples[:100] if quick else samples


def load_voc_samples(root: Path, quick: bool) -> List[ImageSample]:
    image_dir = root / "JPEGImages"
    annotation_dir = root / "Annotations"
    split_path = root / "ImageSets" / "Main" / "val.txt"
    if not image_dir.exists() or not annotation_dir.exists():
        raise FileNotFoundError(f"PASCAL VOC 2012 layout not found under {root}")

    if split_path.exists():
        image_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        image_ids = [path.stem for path in sorted(annotation_dir.glob("*.xml"))]

    samples = []
    for image_id in image_ids:
        image_path = image_dir / f"{image_id}.jpg"
        annotation_path = annotation_dir / f"{image_id}.xml"
        if not image_path.exists() or not annotation_path.exists():
            continue
        annotations = parse_voc_annotation(annotation_path)
        samples.append(ImageSample("voc2012", image_id, image_path, annotations))

    return samples[:100] if quick else samples


def parse_voc_annotation(path: Path) -> List[Dict[str, Any]]:
    root = ET.parse(path).getroot()
    annotations = []
    for obj in root.findall("object"):
        difficult = int(obj.findtext("difficult", default="0"))
        if difficult:
            continue
        class_name = obj.findtext("name", default="")
        class_name = VOC_TO_COCO.get(class_name, class_name)
        box = obj.find("bndbox")
        if box is None:
            continue
        annotations.append({
            "class_name": class_name,
            "bbox": [
                float(box.findtext("xmin", default="0")),
                float(box.findtext("ymin", default="0")),
                float(box.findtext("xmax", default="0")),
                float(box.findtext("ymax", default="0")),
            ],
        })
    return annotations


def evaluate_model(
    service: DetectionService,
    model_id: str,
    samples: List[ImageSample],
    confidence: float,
) -> ModelEval:
    process = psutil.Process()
    peak_ram_mb = process.memory_info().rss / (1024**2)
    peak_vram_mb = 0.0
    predictions_by_image: Dict[str, List[Dict[str, Any]]] = {}
    inference_ms_by_image: Dict[str, float] = {}

    torch = maybe_torch()
    if torch and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for sample in tqdm(samples, desc=f"{model_id} on {samples[0].dataset}", leave=False):
        image = load_image(sample.image_path)
        engine = service.get_engine(model_id)
        detections, inference_ms = engine.detect(
            image,
            confidence_threshold=confidence,
            iou_threshold=0.45,
            max_detections=100,
        )
        predictions_by_image[sample.image_id] = detections
        inference_ms_by_image[sample.image_id] = inference_ms
        peak_ram_mb = max(peak_ram_mb, process.memory_info().rss / (1024**2))
        if torch and torch.cuda.is_available():
            peak_vram_mb = max(peak_vram_mb, torch.cuda.max_memory_allocated() / (1024**2))

    metrics = compute_dataset_metrics(samples, predictions_by_image)
    timing_values = list(inference_ms_by_image.values())[:100]
    metrics.update({
        "inference_time_ms": float(np.mean(timing_values)) if timing_values else 0.0,
        "peak_ram_mb": peak_ram_mb,
        "gpu_vram_mb": peak_vram_mb,
    })

    per_image_ap = {
        sample.image_id: compute_image_ap(sample.annotations, predictions_by_image.get(sample.image_id, []), IOU_THRESHOLDS)
        for sample in samples
    }
    return ModelEval(metrics, per_image_ap, inference_ms_by_image, predictions_by_image)


def compute_dataset_metrics(
    samples: List[ImageSample],
    predictions_by_image: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, float]:
    aps_by_threshold = []
    for threshold in IOU_THRESHOLDS:
        aps_by_threshold.append(compute_map_at_threshold(samples, predictions_by_image, threshold))

    recalls = [
        compute_image_recall(sample.annotations, predictions_by_image.get(sample.image_id, []), 0.5, max_detections=100)
        for sample in samples
        if sample.annotations
    ]

    return {
        "mAP@0.5": aps_by_threshold[0],
        "mAP@0.5:0.95": float(np.mean(aps_by_threshold)) if aps_by_threshold else 0.0,
        "AR@100": float(np.mean(recalls)) if recalls else 0.0,
    }


def compute_map_at_threshold(
    samples: List[ImageSample],
    predictions_by_image: Dict[str, List[Dict[str, Any]]],
    iou_threshold: float,
) -> float:
    classes = sorted({
        ann["class_name"]
        for sample in samples
        for ann in sample.annotations
    })
    class_aps = []
    for class_name in classes:
        gt_by_image = {
            sample.image_id: [ann for ann in sample.annotations if ann["class_name"] == class_name]
            for sample in samples
        }
        total_gt = sum(len(items) for items in gt_by_image.values())
        if total_gt == 0:
            continue

        predictions = []
        for sample in samples:
            for det in predictions_by_image.get(sample.image_id, []):
                if det.get("class_name") == class_name:
                    predictions.append((sample.image_id, float(det.get("confidence", 0)), detection_bbox(det)))
        predictions.sort(key=lambda item: item[1], reverse=True)

        matched = {image_id: set() for image_id in gt_by_image}
        true_positive = np.zeros(len(predictions), dtype=np.float32)
        false_positive = np.zeros(len(predictions), dtype=np.float32)

        for index, (image_id, _, pred_box) in enumerate(predictions):
            gt_items = gt_by_image.get(image_id, [])
            best_iou = 0.0
            best_gt_index = -1
            for gt_index, gt in enumerate(gt_items):
                if gt_index in matched[image_id]:
                    continue
                overlap = iou(pred_box, gt["bbox"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt_index = gt_index

            if best_iou >= iou_threshold and best_gt_index >= 0:
                true_positive[index] = 1
                matched[image_id].add(best_gt_index)
            else:
                false_positive[index] = 1

        if len(predictions) == 0:
            class_aps.append(0.0)
            continue

        tp_cum = np.cumsum(true_positive)
        fp_cum = np.cumsum(false_positive)
        recall = tp_cum / max(total_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        class_aps.append(voc_ap(recall, precision))

    return float(np.mean(class_aps)) if class_aps else 0.0


def compute_image_ap(
    annotations: List[Dict[str, Any]],
    detections: List[Dict[str, Any]],
    thresholds: Iterable[float],
) -> float:
    if not annotations:
        return 0.0
    scores = []
    sample = ImageSample("image", "image", Path(), annotations)
    for threshold in thresholds:
        scores.append(compute_map_at_threshold([sample], {"image": detections}, threshold))
    return float(np.mean(scores)) if scores else 0.0


def compute_image_recall(
    annotations: List[Dict[str, Any]],
    detections: List[Dict[str, Any]],
    iou_threshold: float,
    max_detections: int,
) -> float:
    if not annotations:
        return 0.0
    detections = sorted(detections, key=lambda det: det.get("confidence", 0), reverse=True)[:max_detections]
    matched = set()
    for det in detections:
        pred_box = detection_bbox(det)
        for index, ann in enumerate(annotations):
            if index in matched or det.get("class_name") != ann["class_name"]:
                continue
            if iou(pred_box, ann["bbox"]) >= iou_threshold:
                matched.add(index)
                break
    return len(matched) / len(annotations)


def detection_bbox(det: Dict[str, Any]) -> List[float]:
    box = det["bbox"]
    if isinstance(box, dict):
        return [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
    return [float(value) for value in box]


def iou(box_a: List[float], box_b: List[float]) -> float:
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    inter_width = max(0.0, x_right - x_left)
    inter_height = max(0.0, y_bottom - y_top)
    intersection = inter_width * inter_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    change_indices = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[change_indices + 1] - recall[change_indices]) * precision[change_indices + 1]))


def run_adaptive_comparison(
    service: DetectionService,
    samples: List[ImageSample],
    evals: Dict[str, ModelEval],
) -> List[Dict[str, Any]]:
    router = get_adaptive_switcher()
    baseline_scores = []
    oracle_scores = []
    ours_scores = []
    weights = []

    for sample in tqdm(samples, desc=f"Adaptive routing on {samples[0].dataset}", leave=False):
        weight = max(1, len(sample.annotations))
        weights.append(weight)
        baseline_scores.append(evals["yolov8n"].per_image_ap.get(sample.image_id, 0.0))
        oracle_scores.append(max(model_eval.per_image_ap.get(sample.image_id, 0.0) for model_eval in evals.values()))

        image = load_image(sample.image_path)
        routed_model = router.predict(image)["predicted_model"]
        if routed_model not in evals:
            routed_model = "yolov8n"
        ours_scores.append(evals[routed_model].per_image_ap.get(sample.image_id, 0.0))

    return [
        {
            "dataset": samples[0].dataset,
            "comparison": "baseline_yolov8n",
            "weighted_average_mAP": weighted_average(baseline_scores, weights),
        },
        {
            "dataset": samples[0].dataset,
            "comparison": "oracle_best_per_image",
            "weighted_average_mAP": weighted_average(oracle_scores, weights),
        },
        {
            "dataset": samples[0].dataset,
            "comparison": "ours_meta_router",
            "weighted_average_mAP": weighted_average(ours_scores, weights),
        },
    ]


def weighted_average(values: List[float], weights: List[float]) -> float:
    total_weight = sum(weights)
    return float(sum(value * weight for value, weight in zip(values, weights)) / total_weight) if total_weight else 0.0


def maybe_torch() -> Optional[Any]:
    try:
        import torch

        return torch
    except Exception:
        return None


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    service = DetectionService()
    datasets = [
        ("coco_val2017", load_coco_samples(Path(args.coco_val2017), args.quick)),
        ("voc2012", load_voc_samples(Path(args.voc2012), args.quick)),
    ]

    model_rows = []
    adaptive_rows = []

    for dataset_name, samples in datasets:
        if not samples:
            raise ValueError(f"No samples found for {dataset_name}")

        evals: Dict[str, ModelEval] = {}
        for model_id in tqdm(EXPERIMENT_MODELS, desc=f"Models on {dataset_name}"):
            model_eval = evaluate_model(service, model_id, samples, args.confidence)
            evals[model_id] = model_eval
            row = {
                "result_type": "model_comparison",
                "dataset": dataset_name,
                "model_id": model_id,
                **model_eval.metrics,
            }
            model_rows.append(row)

        adaptive_rows.extend(run_adaptive_comparison(service, samples, evals))

    all_rows = model_rows + [
        {
            "result_type": "adaptive_routing",
            "dataset": row["dataset"],
            "model_id": row["comparison"],
            "mAP@0.5": np.nan,
            "mAP@0.5:0.95": row["weighted_average_mAP"],
            "AR@100": np.nan,
            "inference_time_ms": np.nan,
            "peak_ram_mb": np.nan,
            "gpu_vram_mb": np.nan,
        }
        for row in adaptive_rows
    ]
    results = pd.DataFrame(all_rows)
    results.to_csv(RESULTS_CSV, index=False)

    save_tables(pd.DataFrame(model_rows), pd.DataFrame(adaptive_rows))
    print(f"Saved full results to {RESULTS_CSV}")
    print(f"Saved model comparison table to {TABLE1_TEX}")
    print(f"Saved adaptive routing table to {TABLE2_TEX}")


def save_tables(model_df: pd.DataFrame, adaptive_df: pd.DataFrame) -> None:
    model_table = model_df.copy()
    metric_cols = ["mAP@0.5", "mAP@0.5:0.95", "AR@100", "inference_time_ms", "peak_ram_mb", "gpu_vram_mb"]
    for col in metric_cols:
        model_table[col] = model_table[col].astype(float).round(4 if "mAP" in col or "AR" in col else 2)
    model_table = model_table[["dataset", "model_id", *metric_cols]]
    model_table.to_latex(TABLE1_TEX, index=False, escape=False, caption="Model comparison on COCO val2017 and PASCAL VOC 2012.", label="tab:model_comparison")

    adaptive_table = adaptive_df.copy()
    adaptive_table["weighted_average_mAP"] = adaptive_table["weighted_average_mAP"].astype(float).round(4)
    adaptive_table = adaptive_table[["dataset", "comparison", "weighted_average_mAP"]]
    adaptive_table.to_latex(TABLE2_TEX, index=False, escape=False, caption="Adaptive routing comparison.", label="tab:adaptive_routing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full LenScope benchmark experiments")
    parser.add_argument("--coco-val2017", required=True, help="Path to COCO root containing val2017 and annotations")
    parser.add_argument("--voc2012", required=True, help="Path to PASCAL VOC 2012 root")
    parser.add_argument("--quick", action="store_true", help="Run only 100 images from each dataset")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
