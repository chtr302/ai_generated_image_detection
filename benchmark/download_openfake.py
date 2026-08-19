from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LABELS = ("real", "fake")
LABEL_KEYS = ("label", "class", "category", "target", "is_fake", "fake", "type")
IMAGE_KEYS = ("image", "jpg", "jpeg", "png")
FIELDNAMES = (
    "sample_id",
    "label",
    "relative_path",
    "source_dataset",
    "config",
    "split",
    "model",
    "type",
    "release_date",
    "prompt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the OpenFake benchmark subset from Hugging Face.")
    parser.add_argument("--dataset", default="ComplexDataLab/OpenFake")
    parser.add_argument("--config", default="core")
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", default="", help="Optional Hugging Face dataset revision, tag, or commit hash.")
    parser.add_argument("--output-root", default="benchmark/data/openfake_1k")
    parser.add_argument("--target-per-label", type=int, default=500)
    parser.add_argument("--max-examples", type=int, default=0, help="Stop after this many streamed rows; 0 means no limit.")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing benchmark data folder.")
    return parser.parse_args()


def normalize_label(value: Any, key: str = "") -> str | None:
    if isinstance(value, bool):
        return "fake" if value else "real"
    if isinstance(value, int):
        return "fake" if value == 1 else "real" if value == 0 else None
    if value is None:
        return None

    text = str(value).lower().strip()
    if text in {"real", "human", "natural", "authentic", "0"}:
        return "real"
    if text in {"fake", "ai", "synthetic", "generated", "machine", "1"}:
        return "fake"
    if key == "type" and text in {"image", "text-to-image", "t2i"}:
        return "fake"
    return None


def row_label(row: dict[str, Any]) -> str | None:
    for key in LABEL_KEYS:
        if key in row:
            label = normalize_label(row[key], key)
            if label:
                return label
    return None


def image_ext_from_bytes(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return "png"


def image_ext_from_pil(image: Any) -> str:
    image_format = (getattr(image, "format", "") or "").lower()
    if image_format == "jpeg":
        return "jpg"
    if image_format in {"jpg", "png", "webp"}:
        return image_format
    return "png"


def get_image_value(row: dict[str, Any]) -> Any:
    for key in IMAGE_KEYS:
        value = row.get(key)
        if value is not None:
            return value
    for value in row.values():
        if isinstance(value, (bytes, bytearray)):
            return value
        if hasattr(value, "save") and hasattr(value, "convert"):
            return value
    raise ValueError(f"Cannot find an image column. Available columns: {sorted(row.keys())}")


def save_image(value: Any, output_stem: Path) -> str:
    from PIL import Image

    if isinstance(value, dict):
        data = value.get("bytes")
        if data:
            ext = image_ext_from_bytes(bytes(data))
            path = output_stem.with_suffix(f".{ext}")
            path.write_bytes(bytes(data))
            return path.name

        source_path = value.get("path")
        if source_path:
            with Image.open(source_path) as image:
                return save_image(image, output_stem)

    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        ext = image_ext_from_bytes(data)
        path = output_stem.with_suffix(f".{ext}")
        path.write_bytes(data)
        return path.name

    if hasattr(value, "save") and hasattr(value, "convert"):
        ext = image_ext_from_pil(value)
        path = output_stem.with_suffix(f".{ext}")
        image = value.convert("RGB") if ext in {"jpg", "jpeg"} else value
        save_format = "JPEG" if ext == "jpg" else ext.upper()
        image.save(path, format=save_format)
        return path.name

    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def manifest_row(
    row: dict[str, Any],
    sample_id: str,
    label: str,
    relative_path: str,
    dataset: str,
    config: str,
    split: str,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "label": label,
        "relative_path": relative_path,
        "source_dataset": dataset,
        "config": config,
        "split": split,
        "model": clean_scalar(row.get("model") or row.get("generator") or row.get("source_model")),
        "type": clean_scalar(row.get("type")),
        "release_date": clean_scalar(row.get("release_date") or row.get("date")),
        "prompt": clean_scalar(row.get("prompt") or row.get("caption")),
    }


def write_manifest(output_root: Path, rows: list[dict[str, str]]) -> None:
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(output_root: Path, dataset: str, config: str, split: str, revision: str, target_per_label: int) -> None:
    revision_line = f"Revision: {revision}\n" if revision else ""
    text = f"""OpenFake 1k external benchmark subset

Source dataset: {dataset}
Config: {config}
Split: {split}
{revision_line}Target: {target_per_label} real + {target_per_label} fake = {target_per_label * 2} images
Sampling: first matching rows encountered from the Hugging Face streaming dataset.
Images are saved from Hugging Face dataset bytes when bytes are available.
Labels: real, fake.

Files:
- manifest.csv: main benchmark manifest
- manifest.jsonl: JSONL copy of the manifest
- images/real: real images
- images/fake: fake images
"""
    (output_root / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    if output_root.exists() and not args.force:
        manifest_path = output_root / "manifest.csv"
        if manifest_path.exists():
            print(f"OpenFake data already exists: {output_root}")
            print("Use --force to overwrite files in this folder.")
            return
        raise FileExistsError(f"{output_root} already exists but has no manifest.csv. Use --force to overwrite files.")

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Missing dependency. Install it with: pip install datasets pillow tqdm") from exc

    from tqdm.auto import tqdm

    for label in LABELS:
        (output_root / "images" / label).mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, Any] = {"split": args.split, "streaming": True}
    if args.revision:
        load_kwargs["revision"] = args.revision
    dataset = load_dataset(args.dataset, args.config, **load_kwargs)
    counts = {label: 0 for label in LABELS}
    rows: list[dict[str, str]] = []

    progress = tqdm(desc="Downloading OpenFake", total=args.target_per_label * len(LABELS))
    for index, source_row in enumerate(dataset):
        if args.max_examples and index >= args.max_examples:
            break

        row = dict(source_row)
        label = row_label(row)
        if label not in LABELS or counts[label] >= args.target_per_label:
            continue

        counts[label] += 1
        sample_id = f"openfake_{args.config}_{args.split}_{label}_{counts[label]:06d}"
        filename = save_image(get_image_value(row), output_root / "images" / label / sample_id)
        relative_path = str(Path("images") / label / filename).replace("\\", "/")
        rows.append(manifest_row(row, sample_id, label, relative_path, args.dataset, args.config, args.split))
        progress.update(1)

        if all(count >= args.target_per_label for count in counts.values()):
            break

    progress.close()

    if any(count < args.target_per_label for count in counts.values()):
        raise RuntimeError(f"Not enough images found. Expected {args.target_per_label} per label, got {counts}.")

    rows.sort(key=lambda item: item["sample_id"])
    write_manifest(output_root, rows)
    write_readme(output_root, args.dataset, args.config, args.split, args.revision, args.target_per_label)

    print(json.dumps({"output_root": str(output_root), "counts": counts, "manifest_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
