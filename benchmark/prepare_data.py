from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = ("real", "fake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OpenFake data before running the benchmark.")
    parser.add_argument("--benchmark-root", default="benchmark/data/openfake_1k")
    parser.add_argument("--output", default="benchmark/results/openfake_1k/data_check.json")
    parser.add_argument("--limit-per-label", type=int, default=250)
    return parser.parse_args()


def load_manifest(benchmark_root: Path) -> list[dict[str, str]]:
    manifest_path = benchmark_root / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def validate_rows(benchmark_root: Path, rows: list[dict[str, str]], limit_per_label: int) -> dict[str, object]:
    label_counts = {label: 0 for label in LABELS}
    missing_files: list[str] = []

    for row in rows:
        label = row["label"].lower().strip()
        if label not in label_counts:
            raise ValueError(f"Invalid label: {row['label']}")

        label_counts[label] += 1
        image_path = benchmark_root / row["relative_path"]
        if not image_path.exists():
            missing_files.append(str(image_path))

    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} images, examples: {missing_files[:3]}")
    if any(count < limit_per_label for count in label_counts.values()):
        raise ValueError(f"Need at least {limit_per_label} images per label, found {label_counts}")

    return {
        "benchmark_root": str(benchmark_root),
        "full_manifest_images": len(rows),
        "full_manifest_label_counts": label_counts,
        "benchmark_images": limit_per_label * 2,
        "benchmark_label_counts": {label: limit_per_label for label in LABELS},
        "is_balanced": True,
        "missing_files": 0,
    }


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    output_path = Path(args.output)

    rows = load_manifest(benchmark_root)
    report = validate_rows(benchmark_root, rows, args.limit_per_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Data check: {output_path}")


if __name__ == "__main__":
    main()
