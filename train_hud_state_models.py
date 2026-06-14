# train the data

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SIZE = (24, 24)
IGNORED_LABELS = {"uncertain"}


def extract_features(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(
        image.convert("RGB").resize(FEATURE_SIZE, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    gray = (
        rgb[:, :, 0] * 0.299
        + rgb[:, :, 1] * 0.587
        + rgb[:, :, 2] * 0.114
    )
    gradient_x = np.zeros_like(gray)
    gradient_y = np.zeros_like(gray)
    gradient_x[:, 1:] = np.abs(np.diff(gray, axis=1))
    gradient_y[1:, :] = np.abs(np.diff(gray, axis=0))

    histograms = []
    for channel in range(3):
        histogram, _ = np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0))
        histograms.append(histogram.astype(np.float32) / rgb[:, :, channel].size)

    return np.concatenate(
        [
            rgb.reshape(-1),
            gray.reshape(-1),
            gradient_x.reshape(-1),
            gradient_y.reshape(-1),
            *histograms,
        ]
    )


def load_samples(
    annotations_path: Path,
    excluded_regions: set[str] | None = None,
) -> dict[str, list[tuple[np.ndarray, str]]]:
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations = (
        payload.get("annotations", []) if isinstance(payload, dict) else payload
    )
    dataset_root = annotations_path.parent
    samples: dict[str, list[tuple[np.ndarray, str]]] = defaultdict(list)
    excluded = {name.casefold() for name in (excluded_regions or set())}

    for entry in annotations:
        region = str(entry.get("region", ""))
        if region.casefold() in excluded:
            continue
        label = str(entry.get("label", ""))
        if not label or label in IGNORED_LABELS:
            continue
        crop_path = dataset_root / str(entry["crop"])
        if not crop_path.is_file():
            print(f"Skipping missing crop: {crop_path}")
            continue
        with Image.open(crop_path) as image:
            features = extract_features(image)
        samples[region].append((features, label))
    return samples


def build_classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            C=4.0,
            random_state=42,
            max_iter=2000,
        ),
    )


def train_models(
    samples: dict[str, list[tuple[np.ndarray, str]]],
) -> tuple[dict[str, object], dict[str, object]]:
    models: dict[str, object] = {}
    report: dict[str, object] = {}

    for region, region_samples in sorted(samples.items()):
        features = np.stack([sample[0] for sample in region_samples])
        labels = np.asarray([sample[1] for sample in region_samples])
        counts = Counter(labels)
        if len(counts) < 2:
            print(f"Skipping {region}: at least two labels are required.")
            continue

        validation_accuracy: float | None = None
        minimum_count = min(counts.values())
        if len(labels) >= 20 and minimum_count >= 2:
            test_size = max(len(counts), round(len(labels) * 0.25))
            train_x, test_x, train_y, test_y = train_test_split(
                features,
                labels,
                test_size=test_size,
                random_state=42,
                stratify=labels,
            )
            validation_model = build_classifier()
            validation_model.fit(train_x, train_y)
            validation_accuracy = accuracy_score(
                test_y, validation_model.predict(test_x)
            )

        model = build_classifier()
        model.fit(features, labels)
        models[region] = model
        report[region] = {
            "samples": len(labels),
            "labels": dict(sorted(counts.items())),
            "validation_accuracy": validation_accuracy,
        }

        accuracy_text = (
            f"{validation_accuracy:.1%}"
            if validation_accuracy is not None
            else "not enough samples"
        )
        print(
            f"{region}: {len(labels)} samples, "
            f"{dict(sorted(counts.items()))}, validation={accuracy_text}"
        )

    if not models:
        raise RuntimeError("No region had enough labeled data to train a model.")
    return models, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train per-region HUD state classifiers."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("training-data/annotations.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training-data/hud-state-models.joblib"),
    )
    parser.add_argument(
        "--exclude-region",
        action="append",
        default=[],
        help="Region to omit from training. May be specified more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded_regions = {name.strip() for name in args.exclude_region if name.strip()}
    if excluded_regions:
        print("Excluding regions from training: " + ", ".join(sorted(excluded_regions)))
    samples = load_samples(args.annotations, excluded_regions)
    models, report = train_models(samples)
    bundle = {
        "format_version": 1,
        "feature_size": FEATURE_SIZE,
        "ignored_labels": sorted(IGNORED_LABELS),
        "excluded_regions": sorted(excluded_regions),
        "models": models,
        "report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(models)} models to {args.output}")
    print(f"Saved training report to {report_path}")


if __name__ == "__main__":
    main()
