"""Dataset helpers for real-vs-AI image classification."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except ImportError:  # pragma: no cover - torch is optional until training time
    class _TorchIterableDataset:  # type: ignore[no-redef]
        pass


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

LABELS = {
    "0": 0,
    "real": 0,
    "authentic": 0,
    "human": 0,
    "natural": 0,
    "photo": 0,
    "original": 0,
    "1": 1,
    "ai": 1,
    "ai-generated": 1,
    "aigenerated": 1,
    "generated": 1,
    "synthetic": 1,
    "fake": 1,
    "gan": 1,
    "diffusion": 1,
}

SPLITS = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    split: str | None = None
    source: str | None = None


def normalize_label(label: str | int) -> int:
    """Normalize labels to binary ids: real=0, AI-generated=1."""

    if isinstance(label, int):
        if label in (0, 1):
            return label
        raise ValueError(f"Invalid numeric label: {label}")

    key = label.strip().lower().replace("_", "-")
    if key not in LABELS:
        raise ValueError(f"Invalid label: {label}. Use real/0 or ai/1.")
    return LABELS[key]


def discover_image_records(root: str | Path) -> list[ImageRecord]:
    """Scan a local folder and infer label/split from folder names."""

    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    records: list[ImageRecord] = []
    skipped: list[Path] = []

    for image_path in sorted(root.rglob("*")):
        if not _is_image(image_path):
            continue

        folders = image_path.relative_to(root).parts[:-1]
        label = _find_label(folders)
        if label is None:
            skipped.append(image_path)
            continue

        records.append(
            ImageRecord(
                path=image_path,
                label=label,
                split=_find_split(folders),
                source=_find_source(folders),
            )
        )

    if skipped:
        examples = ", ".join(str(path) for path in skipped[:3])
        raise ValueError(
            "Cannot infer labels for some images. Put images under real/ai/generated/fake folders. "
            f"Examples: {examples}"
        )
    if not records:
        raise ValueError(f"No supported image files found in: {root}")

    return records


def split_records(
    records: Sequence[ImageRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[ImageRecord]:
    """Split records into train/val/test while keeping label balance."""

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    rng = random.Random(seed)
    records_by_label: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_label[record.label].append(record)

    result: list[ImageRecord] = []
    for group in records_by_label.values():
        group = list(group)
        rng.shuffle(group)

        train_end = round(len(group) * train_ratio)
        val_end = train_end + round(len(group) * val_ratio)

        for index, record in enumerate(group):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"
            result.append(replace(record, split=split))

    return sorted(result, key=lambda record: str(record.path))


class ImageDataset:
    """Map-style PyTorch dataset for local image files."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Callable | None = None,
        return_metadata: bool = True,
    ) -> None:
        if not records:
            raise ValueError("Dataset needs at least one image")
        self.records = list(records)
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Install Pillow to read images: pip install pillow") from exc

        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        if not self.return_metadata:
            return image, record.label

        metadata = {
            "path": str(record.path),
            "split": record.split or "",
            "source": record.source or "",
        }
        return image, record.label, metadata


class HuggingFaceImageDataset:
    """Map-style adapter for Defactify Hugging Face rows."""

    def __init__(
        self,
        dataset,
        image_column: str = "Image",
        label_column: str = "Label_A",
        caption_column: str = "Caption",
        source_label_column: str = "Label_B",
        transform: Callable | None = None,
        return_metadata: bool = True,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("Hugging Face dataset split needs at least one image")
        self.dataset = dataset
        self.image_column = image_column
        self.label_column = label_column
        self.caption_column = caption_column
        self.source_label_column = source_label_column
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return _format_hf_row(
            row=self.dataset[index],
            index=index,
            image_column=self.image_column,
            label_column=self.label_column,
            caption_column=self.caption_column,
            source_label_column=self.source_label_column,
            transform=self.transform,
            return_metadata=self.return_metadata,
        )


class StreamingHuggingFaceImageDataset(_TorchIterableDataset):
    """Streaming adapter for Defactify rows without downloading the full dataset."""

    def __init__(
        self,
        dataset,
        image_column: str = "Image",
        label_column: str = "Label_A",
        caption_column: str = "Caption",
        source_label_column: str = "Label_B",
        transform: Callable | None = None,
        return_metadata: bool = True,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.image_column = image_column
        self.label_column = label_column
        self.caption_column = caption_column
        self.source_label_column = source_label_column
        self.transform = transform
        self.return_metadata = return_metadata

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator:
        dataset = self.dataset
        worker_info = _get_torch_worker_info()
        shard_applied = False
        if worker_info is not None and hasattr(dataset, "shard"):
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            shard_applied = True

        output_index = 0
        for input_index, row in enumerate(dataset):
            if worker_info is not None and not shard_applied and input_index % worker_info.num_workers != worker_info.id:
                continue
            yield _format_hf_row(
                row=row,
                index=output_index,
                image_column=self.image_column,
                label_column=self.label_column,
                caption_column=self.caption_column,
                source_label_column=self.source_label_column,
                transform=self.transform,
                return_metadata=self.return_metadata,
            )
            output_index += 1

def _to_rgb_pil(value: Any):
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Install Pillow to read images: pip install pillow") from exc

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    if hasattr(value, "__array__"):
        return Image.fromarray(value).convert("RGB")
    raise TypeError(f"Cannot read image value from Hugging Face row: {type(value)!r}")


def _format_hf_row(
    row: dict[str, Any],
    index: int,
    image_column: str,
    label_column: str,
    caption_column: str,
    source_label_column: str,
    transform: Callable | None,
    return_metadata: bool,
):
    image = _to_rgb_pil(row[image_column])
    label = normalize_label(int(row[label_column]))

    if transform:
        image = transform(image)

    if not return_metadata:
        return image, label

    metadata = {
        "path": str(row.get("image_id", index)),
        "split": str(row.get("split", "")),
        "source": str(row.get(source_label_column, "")),
        "caption": str(row.get(caption_column, "")),
    }
    return image, label, metadata


def _get_torch_worker_info():
    try:
        from torch.utils.data import get_worker_info
    except ImportError:
        return None
    return get_worker_info()


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _find_label(folders: Sequence[str]) -> int | None:
    for folder in reversed(folders):
        key = folder.strip().lower().replace("_", "-")
        if key in LABELS:
            return LABELS[key]
    return None


def _find_split(folders: Sequence[str]) -> str | None:
    for folder in folders:
        split = _normalize_split(folder, allow_unknown=True)
        if split:
            return split
    return None


def _find_source(folders: Sequence[str]) -> str | None:
    for folder in reversed(folders):
        key = folder.strip().lower().replace("_", "-")
        if key and key not in LABELS and key not in SPLITS:
            return folder
    return None


def _normalize_split(split: str | None, allow_unknown: bool = False) -> str | None:
    key = (split or "").strip().lower().replace("_", "-")
    if not key:
        return None
    if key in SPLITS:
        return SPLITS[key]
    if allow_unknown:
        return None
    raise ValueError(f"Invalid split: {split}. Use train/val/test.")


