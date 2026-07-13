"""Dataset cho bài toán phân loại ảnh thật và ảnh AI."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# Các tên label khác nhau được quy về 2 lớp: real=0, ai=1.
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

# Các tên split hợp lệ được chuẩn hóa về train/val/test.
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
    # Một ảnh sau khi được chuẩn hóa thông tin.
    path: Path
    label: int
    split: str | None = None
    source: str | None = None


def normalize_label(label: str | int) -> int:
    """Đưa label về dạng số: real=0, ai=1."""

    # Label số chỉ được phép là 0 hoặc 1.
    if isinstance(label, int):
        if label in (0, 1):
            return label
        raise ValueError(f"Label số không hợp lệ: {label}")

    # Label chữ được đưa về dạng thống nhất trước khi tra bảng LABELS.
    key = label.strip().lower().replace("_", "-")
    if key not in LABELS:
        raise ValueError(f"Label không hợp lệ: {label}. Dùng real/0 hoặc ai/1.")
    return LABELS[key]


def discover_image_records(root: str | Path) -> list[ImageRecord]:
    """Quét folder dataset và suy ra label/split từ tên thư mục."""

    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset root: {root}")

    records: list[ImageRecord] = []
    skipped: list[Path] = []

    for image_path in sorted(root.rglob("*")):
        # Chỉ xử lý các file ảnh có extension được hỗ trợ.
        if not _is_image(image_path):
            continue

        folders = image_path.relative_to(root).parts[:-1]
        # Label được suy ra từ tên folder cha như real, ai, generated, fake.
        label = _find_label(folders)
        if label is None:
            skipped.append(image_path)
            continue

        records.append(
            ImageRecord(
                path=image_path,
                label=label,
                # Nếu folder có train/val/test thì giữ lại, nếu không thì để None.
                split=_find_split(folders),
                # Source là folder mô tả nguồn dữ liệu nếu có, ví dụ sdxl/midjourney.
                source=_find_source(folders),
            )
        )

    if skipped:
        examples = ", ".join(str(path) for path in skipped[:3])
        raise ValueError(
            "Không suy ra được label từ một số ảnh. "
            f"Hãy đặt ảnh trong folder real/ai/generated/fake. Ví dụ: {examples}"
        )
    if not records:
        raise ValueError(f"Không tìm thấy ảnh hợp lệ trong: {root}")

    return records


def split_records(
    records: Sequence[ImageRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[ImageRecord]:
    """Chia dataset thành train/val/test theo từng label."""

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Tổng tỷ lệ split phải lớn hơn 0")

    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    rng = random.Random(seed)
    records_by_label: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        # Chia riêng theo label để train/val/test không lệch lớp quá nhiều.
        records_by_label[record.label].append(record)

    result: list[ImageRecord] = []
    for group in records_by_label.values():
        group = list(group)
        rng.shuffle(group)

        train_end = round(len(group) * train_ratio)
        val_end = train_end + round(len(group) * val_ratio)

        for index, record in enumerate(group):
            # Gán split dựa trên vị trí sau khi shuffle.
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"
            result.append(replace(record, split=split))

    return sorted(result, key=lambda record: str(record.path))


class ImageDataset:
    """Dataset trả về image, label và metadata cho PyTorch."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Callable | None = None,
        return_metadata: bool = True,
    ) -> None:
        if not records:
            raise ValueError("Dataset cần ít nhất một ảnh")
        self.records = list(records)
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Cài Pillow để đọc ảnh: pip install pillow") from exc

        # Ảnh chỉ được mở khi DataLoader cần lấy item.
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")

        # Transform biến ảnh PIL thành tensor đã chuẩn hóa.
        if self.transform:
            image = self.transform(image)

        if not self.return_metadata:
            return image, record.label

        # Metadata giúp truy vết ảnh khi debug hoặc đánh giá lỗi.
        metadata = {
            "path": str(record.path),
            "split": record.split or "",
            "source": record.source or "",
        }
        return image, record.label, metadata


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
    raise ValueError(f"Split không hợp lệ: {split}. Dùng train/val/test.")
