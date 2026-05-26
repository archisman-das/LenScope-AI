"""
Train, evaluate, and run the LenScope meta-router.

The router learns from backend/data/meta_training_data.jsonl rows produced by
MetaDatasetCollector. Supervised training requires rows with non-null mAP
scores, ideally from benchmark/evaluation runs that test multiple models on the
same scene.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
DATA_PATH = BACKEND_ROOT / "data" / "meta_training_data.jsonl"
SKLEARN_MODEL_PATH = BACKEND_ROOT / "weights" / "meta_router.pkl"
TORCH_MODEL_PATH = BACKEND_ROOT / "weights" / "meta_router.pt"
CONFUSION_PATH = PROJECT_ROOT / "benchmark" / "results" / "meta_router_confusion.png"

RESOLUTION_BUCKETS = ["small", "medium", "large"]
SCENE_TYPES = ["indoor", "outdoor", "crowd", "vehicle", "night"]
FEATURE_DIM = 13


@dataclass
class RouterDataset:
    features: np.ndarray
    labels: np.ndarray
    label_encoder: LabelEncoder


def load_jsonl(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Meta training data not found: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping invalid JSON on line {line_no}: {exc}")
    return rows


def get_map_score(row: Dict[str, Any]) -> Optional[float]:
    for key in ("map_score", "mAP", "map", "mAP_score"):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def feature_key(scene_features: Dict[str, Any]) -> str:
    return json.dumps(scene_features, sort_keys=True, separators=(",", ":"))


def best_rows_by_scene(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_scene: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        scene_features = row.get("scene_features")
        model_id = row.get("model_used") or row.get("model_id")
        map_score = get_map_score(row)
        if not scene_features or not model_id or map_score is None:
            continue

        key = feature_key(scene_features)
        current = best_by_scene.get(key)
        if current is None or map_score > get_map_score(current):
            best_by_scene[key] = row

    return list(best_by_scene.values())


def vectorize_scene_features(scene_features: Dict[str, Any]) -> List[float]:
    bucket = scene_features.get("resolution_bucket")
    scene_type = scene_features.get("scene_type")

    vector = [
        float(scene_features.get("mean_brightness", 0.0) or 0.0),
        float(scene_features.get("contrast", 0.0) or 0.0),
        float(scene_features.get("blur_score", 0.0) or 0.0),
        float(scene_features.get("estimated_object_count", 0.0) or 0.0),
        float(scene_features.get("aspect_ratio", 0.0) or 0.0),
    ]
    vector.extend(1.0 if bucket == value else 0.0 for value in RESOLUTION_BUCKETS)
    vector.extend(1.0 if scene_type == value else 0.0 for value in SCENE_TYPES)
    return vector


def build_dataset(rows: Iterable[Dict[str, Any]]) -> RouterDataset:
    from sklearn.preprocessing import LabelEncoder

    labeled_rows = best_rows_by_scene(rows)
    if not labeled_rows:
        raise ValueError("No labeled rows found. Training requires rows with map_score/mAP values.")

    features = np.array(
        [vectorize_scene_features(row["scene_features"]) for row in labeled_rows],
        dtype=np.float32,
    )
    labels = np.array([row.get("model_used") or row.get("model_id") for row in labeled_rows])
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(labels)
    return RouterDataset(features=features, labels=encoded, label_encoder=label_encoder)


def split_dataset(dataset: RouterDataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    if len(dataset.labels) < 2:
        raise ValueError("Need at least 2 labeled scenes to train/evaluate the router.")

    class_counts = np.bincount(dataset.labels)
    stratify = dataset.labels if len(class_counts) > 1 and np.min(class_counts) >= 2 else None
    return train_test_split(
        dataset.features,
        dataset.labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )


def train_random_forest(dataset: RouterDataset) -> RandomForestClassifier:
    from sklearn.ensemble import RandomForestClassifier

    x_train, x_test, y_train, y_test = split_dataset(dataset)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    evaluate_predictions(dataset.label_encoder, y_test, model.predict(x_test))
    save_confusion_matrix(dataset.label_encoder, y_test, model.predict(x_test))
    save_sklearn_model(model, dataset.label_encoder)
    return model


def save_sklearn_model(model: RandomForestClassifier, label_encoder: LabelEncoder) -> None:
    SKLEARN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SKLEARN_MODEL_PATH.open("wb") as file:
        pickle.dump(
            {
                "kind": "sklearn",
                "feature_dim": FEATURE_DIM,
                "resolution_buckets": RESOLUTION_BUCKETS,
                "scene_types": SCENE_TYPES,
                "label_encoder": label_encoder,
                "model": model,
            },
            file,
        )
    print(f"Saved sklearn router to {SKLEARN_MODEL_PATH}")


def train_torch_head(dataset: RouterDataset, epochs: int = 80, batch_size: int = 64) -> Any:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    x_train, x_test, y_train, y_test = split_dataset(dataset)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    model = MetaMobileNetV3Head(FEATURE_DIM, len(dataset.label_encoder.classes_))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64))),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test))
        predictions = logits.argmax(dim=1).numpy()

    evaluate_predictions(dataset.label_encoder, y_test, predictions)
    save_confusion_matrix(dataset.label_encoder, y_test, predictions)
    save_torch_model(model, scaler, dataset.label_encoder)
    return model


class MetaMobileNetV3Head:
    """MobileNetV3-Small classifier head adapted for tabular scene features."""

    def __new__(cls, input_dim: int, num_classes: int) -> Any:
        import torch.nn as nn
        from torchvision.models import mobilenet_v3_small

        backbone = mobilenet_v3_small(weights=None)
        classifier = list(backbone.classifier.children())
        classifier[0] = nn.Linear(input_dim, classifier[0].out_features)
        classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
        return nn.Sequential(*classifier)


def save_torch_model(model: Any, scaler: StandardScaler, label_encoder: LabelEncoder) -> None:
    import torch

    TORCH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "torch",
            "feature_dim": FEATURE_DIM,
            "resolution_buckets": RESOLUTION_BUCKETS,
            "scene_types": SCENE_TYPES,
            "classes": label_encoder.classes_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "state_dict": model.state_dict(),
        },
        TORCH_MODEL_PATH,
    )
    print(f"Saved torch router to {TORCH_MODEL_PATH}")


def evaluate_predictions(label_encoder: LabelEncoder, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import classification_report

    labels = np.arange(len(label_encoder.classes_))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )


def save_confusion_matrix(label_encoder: LabelEncoder, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    labels = np.arange(len(label_encoder.classes_))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    CONFUSION_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(6, len(labels) * 0.65)
    fig, axis = plt.subplots(figsize=(fig_width, fig_width))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=labels,
        yticks=labels,
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ylabel="True model",
        xlabel="Predicted model",
        title="Meta-router Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=160)
    plt.close(fig)
    print(f"Saved confusion matrix to {CONFUSION_PATH}")


def train() -> None:
    dataset = build_dataset(load_jsonl())
    sample_count = len(dataset.labels)
    class_count = len(dataset.label_encoder.classes_)
    print(f"Loaded {sample_count} labeled scenes across {class_count} model classes")

    if class_count < 2:
        raise ValueError("Need at least 2 model classes to train a multi-class router.")

    if sample_count < 500:
        print("Sample count < 500; using RandomForestClassifier fallback")
        train_random_forest(dataset)
    else:
        print("Sample count >= 500; training MobileNetV3-Small classifier head")
        train_torch_head(dataset)


def evaluate() -> None:
    dataset = build_dataset(load_jsonl())
    if SKLEARN_MODEL_PATH.exists():
        bundle = load_sklearn_bundle()
        _, x_test, _, y_test = split_dataset(dataset)
        predictions = bundle["model"].predict(x_test)
        evaluate_predictions(bundle["label_encoder"], y_test, predictions)
        save_confusion_matrix(bundle["label_encoder"], y_test, predictions)
        return

    if TORCH_MODEL_PATH.exists():
        evaluate_torch(dataset)
        return

    raise FileNotFoundError("No trained meta-router found. Run --mode train first.")


def evaluate_torch(dataset: RouterDataset) -> None:
    import torch
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    checkpoint = torch.load(TORCH_MODEL_PATH, map_location="cpu")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(checkpoint["classes"])
    _, x_test, _, y_test = split_dataset(dataset)
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = FEATURE_DIM
    x_test = scaler.transform(x_test).astype(np.float32)

    model = MetaMobileNetV3Head(FEATURE_DIM, len(label_encoder.classes_))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_test)).argmax(dim=1).numpy()

    evaluate_predictions(label_encoder, y_test, predictions)
    save_confusion_matrix(label_encoder, y_test, predictions)


def predict(image_path: Path) -> None:
    scene_features = extract_features_for_image(image_path)
    vector = np.array([vectorize_scene_features(scene_features)], dtype=np.float32)

    if SKLEARN_MODEL_PATH.exists():
        bundle = load_sklearn_bundle()
        prediction = bundle["model"].predict(vector)[0]
        model_id = bundle["label_encoder"].inverse_transform([prediction])[0]
    elif TORCH_MODEL_PATH.exists():
        model_id = predict_torch(vector)
    else:
        raise FileNotFoundError("No trained meta-router found. Run --mode train first.")

    print(json.dumps({"recommended_model": model_id, "scene_features": scene_features}, indent=2))


def extract_features_for_image(image_path: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from backend.app.services.meta_dataset_collector import MetaDatasetCollector
    from backend.app.utils.image import load_image

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(image_path)
    return MetaDatasetCollector()._extract_scene_features(image)


def load_sklearn_bundle() -> Dict[str, Any]:
    with SKLEARN_MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def predict_torch(vector: np.ndarray) -> str:
    import torch
    from sklearn.preprocessing import StandardScaler

    checkpoint = torch.load(TORCH_MODEL_PATH, map_location="cpu")
    classes = np.array(checkpoint["classes"])
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = FEATURE_DIM
    vector = scaler.transform(vector).astype(np.float32)

    model = MetaMobileNetV3Head(FEATURE_DIM, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = int(model(torch.from_numpy(vector)).argmax(dim=1).item())
    return str(classes[prediction])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate/predict with the LenScope meta-router")
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], required=True)
    parser.add_argument("image_path", nargs="?", help="Image path for --mode predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "predict":
        if not args.image_path:
            raise ValueError("predict mode requires an image_path argument")
        predict(Path(args.image_path))


if __name__ == "__main__":
    main()
