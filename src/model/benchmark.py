from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.data.dataloader import DEFAULT_HF_DATASET
from src.model.architectures.detector import build_detector
from src.model.evaluate import evaluate_binary_classifier
from src.model.train import _amp_dtype, build_train_loaders
from src.xai.concept_bottleneck import DEFAULT_CONCEPTS


CORE_METRICS = ["accuracy", "balanced_accuracy", "precision_ai", "recall_ai", "f1_ai", "specificity_real"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark detector on val/test split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-config-name", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--hf-no-streaming", action="store_true")
    parser.add_argument("--hf-shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nec", type=int, default=10)
    parser.add_argument("--concept-count", type=int, default=len(DEFAULT_CONCEPTS))
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--compare-json", default=None, help="Optional JSON list with external model metrics on the same 100 images.")
    return parser.parse_args()


def benchmark_checkpoint(
    checkpoint_path: str | Path,
    split: str = "test",
    max_samples: int = 100,
    data_root: str | Path | None = None,
    hf_dataset_id: str | None = DEFAULT_HF_DATASET,
    hf_config_name: str | None = None,
    hf_cache_dir: str | Path | None = None,
    hf_streaming: bool = True,
    hf_shuffle_buffer: int = 10_000,
    image_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 0,
    nec: int = 10,
    concept_count: int = len(DEFAULT_CONCEPTS),
    amp: str = "fp16",
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector(nec=nec, concept_count=concept_count, image_size=image_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    loaders, _ = build_train_loaders(
        data_root=data_root,
        hf_dataset_id=hf_dataset_id if data_root is None else None,
        hf_config_name=hf_config_name,
        hf_cache_dir=hf_cache_dir,
        hf_streaming=hf_streaming,
        hf_shuffle_buffer=hf_shuffle_buffer,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        distributed=False,
        pin_memory=device.type == "cuda",
    )
    if split not in loaders:
        raise ValueError(f"Dataset does not provide split '{split}'. Available splits: {sorted(loaders)}")

    metrics = evaluate_binary_classifier(
        model=model,
        loader=loaders[split],
        device=device,
        amp_dtype=_amp_dtype(amp),
        max_samples=max_samples,
    )
    return {
        "model": "SFW-SwinCBM",
        "split": split,
        "requested_samples": max_samples,
        "sample_count": int(metrics.get("sample_count", 0)),
        "metrics": {key: round(float(metrics[key]), 4) for key in CORE_METRICS if key in metrics},
    }


def load_comparison_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("models", [])
    if not isinstance(raw, list):
        raise ValueError("compare-json must be a list, or an object with a 'models' list.")
    rows: list[dict[str, Any]] = []
    for item in raw:
        row = dict(item)
        metrics = row.pop("metrics", None)
        if isinstance(metrics, dict):
            row.update(metrics)
        rows.append(row)
    return rows


def format_benchmark_table(own_result: dict[str, Any], comparison_rows: list[dict[str, Any]] | None = None) -> str:
    rows = [_own_row(own_result)] + list(comparison_rows or [])
    header = f"{'model':<28} {'samples':>7} {'acc':>8} {'bal_acc':>8} {'prec_ai':>8} {'rec_ai':>8} {'f1_ai':>8} note"
    lines = ["Benchmark on same 100-image test protocol", "-" * len(header), header]
    for row in rows:
        lines.append(
            f"{str(row.get('model', '-'))[:28]:<28} "
            f"{_format_count(row.get('sample_count', row.get('samples'))):>7} "
            f"{_format_metric(row.get('accuracy')):>8} "
            f"{_format_metric(row.get('balanced_accuracy')):>8} "
            f"{_format_metric(row.get('precision_ai')):>8} "
            f"{_format_metric(row.get('recall_ai')):>8} "
            f"{_format_metric(row.get('f1_ai')):>8} "
            f"{row.get('note', '')}"
        )
    lines.append("-" * len(header))
    return "\n".join(lines)


def _own_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": result["model"],
        "sample_count": result["sample_count"],
        **result["metrics"],
        "note": f"{result['split']} split",
    }


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_count(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root) if args.data_root else None
    result = benchmark_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_samples=args.max_samples,
        data_root=data_root,
        hf_dataset_id=args.hf_dataset,
        hf_config_name=args.hf_config_name,
        hf_cache_dir=args.hf_cache_dir,
        hf_streaming=not args.hf_no_streaming,
        hf_shuffle_buffer=args.hf_shuffle_buffer,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        nec=args.nec,
        concept_count=args.concept_count,
        amp=args.amp,
    )
    comparison_rows = load_comparison_rows(args.compare_json)
    print(format_benchmark_table(result, comparison_rows))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
