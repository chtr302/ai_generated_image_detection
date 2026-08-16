from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


LABEL_TO_INT = {"real": 0, "fake": 1}
INT_TO_LABEL = {0: "real", 1: "fake"}
THRESHOLDS = (0.5, 0.65, 0.8)

RAW_INPUTS = {
    "Hybrid XRayon Physical": "raw_hybrid_xrayon_physical.csv",
    "XRayon RGB Only": "raw_xrayon_rgb_only.csv",
    "UniversalFakeDetect": "raw_universalfakedetect.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3 benchmark models at fixed thresholds.")
    parser.add_argument("--output-root", default="benchmark/results/openfake_1k")
    parser.add_argument("--limit-per-label", type=int, default=250)
    return parser.parse_args()


def read_raw_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    parsed: list[dict[str, Any]] = []
    for row in rows:
        true_label = row["true_label"].lower().strip()
        parsed.append(
            {
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "true_label": true_label,
                "y_true": LABEL_TO_INT[true_label],
                "fake_score": float(row["fake_score"]),
                "source_model": row.get("source_model", ""),
                "source_type": row.get("source_type", ""),
            }
        )
    return parsed


def select_balanced_subset(rows: list[dict[str, Any]], limit_per_label: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label in (0, 1):
        label_rows = [row for row in rows if int(row["y_true"]) == label]
        if len(label_rows) < limit_per_label:
            raise ValueError(f"Need {limit_per_label} rows for label {label}, found {len(label_rows)}")
        if limit_per_label == len(label_rows):
            selected.extend(label_rows)
            continue

        indices = np.linspace(0, len(label_rows) - 1, limit_per_label, dtype=int)
        selected.extend(label_rows[int(index)] for index in indices)

    return sorted(selected, key=lambda row: row["sample_id"])


def arrays(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    return [int(row["y_true"]) for row in rows], [float(row["fake_score"]) for row in rows]


def labels_from_threshold(scores: list[float], threshold: float) -> list[int]:
    return [1 if score >= threshold else 0 for score in scores]


def metrics_at_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    y_true, scores = arrays(rows)
    y_pred = labels_from_threshold(scores, threshold)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "true_real_pred_real": cm[0][0],
        "true_real_pred_fake": cm[0][1],
        "true_fake_pred_real": cm[1][0],
        "true_fake_pred_fake": cm[1][1],
    }


def score_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    y_true, scores = arrays(rows)
    real_scores = np.array([score for label, score in zip(y_true, scores) if label == 0])
    fake_scores = np.array([score for label, score in zip(y_true, scores) if label == 1])
    return {
        "real_mean_score": float(real_scores.mean()),
        "fake_mean_score": float(fake_scores.mean()),
        "real_median_score": float(np.median(real_scores)),
        "fake_median_score": float(np.median(fake_scores)),
    }


def result_row(model_name: str, rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return {
        "model": model_name,
        "mode": f"fixed_{threshold:g}",
        "split": "balanced_500",
        "threshold_source": "fixed",
        "n_images": len(rows),
        **metrics_at_threshold(rows, threshold),
        **score_summary(rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, model_name: str, rows: list[dict[str, Any]]) -> None:
    prediction_rows: list[dict[str, Any]] = []
    for row in rows:
        for threshold in THRESHOLDS:
            pred_int = labels_from_threshold([float(row["fake_score"])], threshold)[0]
            pred_label = INT_TO_LABEL[pred_int]
            prediction_rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "sample_id": row["sample_id"],
                    "relative_path": row["relative_path"],
                    "true_label": row["true_label"],
                    "pred_label": pred_label,
                    "fake_score": row["fake_score"],
                    "correct": int(pred_label == row["true_label"]),
                    "source_model": row.get("source_model", ""),
                    "source_type": row.get("source_type", ""),
                }
            )
    write_csv(path, prediction_rows)


def evaluate_model(model_name: str, raw_path: Path, output_root: Path, limit_per_label: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw prediction file for {model_name}: {raw_path}")

    rows = select_balanced_subset(read_raw_predictions(raw_path), limit_per_label)
    metrics_rows = [result_row(model_name, rows, threshold) for threshold in THRESHOLDS]

    predictions_path = output_root / "predictions" / f"fixed_threshold_predictions_{raw_path.stem.removeprefix('raw_')}.csv"
    write_predictions(predictions_path, model_name, rows)

    return metrics_rows, {
        "raw_predictions": str(raw_path),
        "fixed_threshold_predictions": str(predictions_path),
        "n_images": len(rows),
        "n_real": sum(1 for row in rows if int(row["y_true"]) == 0),
        "n_fake": sum(1 for row in rows if int(row["y_true"]) == 1),
        "thresholds": list(THRESHOLDS),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    prediction_root = output_root / "predictions"

    all_metrics: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "protocol": {
            "models": list(RAW_INPUTS.keys()),
            "thresholds": list(THRESHOLDS),
            "limit_per_label": args.limit_per_label,
            "total_images": args.limit_per_label * 2,
            "threshold_rule": "Each model is evaluated at fixed thresholds 0.5, 0.65, and 0.8. No calibration split is used.",
        },
        "models": {},
    }

    for model_name, filename in RAW_INPUTS.items():
        metrics_rows, model_report = evaluate_model(model_name, prediction_root / filename, output_root, args.limit_per_label)
        all_metrics.extend(metrics_rows)
        report["models"][model_name] = model_report

    write_csv(output_root / "metrics.csv", all_metrics)
    (output_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Metrics: {output_root / 'metrics.csv'}")
    print(f"Report: {output_root / 'report.json'}")


if __name__ == "__main__":
    main()
