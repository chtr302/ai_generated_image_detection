"""Tạo PyTorch Dataset và DataLoader từ folder ảnh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .dataset import (
    ImageDataset,
    ImageRecord,
    discover_image_records,
    split_records,
)
from .transforms import build_eval_transform, build_train_transform


@dataclass(frozen=True)
class DataLoaderConfig:
    # Cấu hình chung để tạo Dataset và DataLoader.
    data_root: str | Path
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    return_metadata: bool = True
    drop_last_train_batch: bool = False


def build_datasets(config: DataLoaderConfig) -> dict[str, ImageDataset]:
    """Tạo dataset cho từng split: train, val, test."""

    # Bước 1: quét danh sách ảnh từ folder dataset.
    records = _load_records(config)
    # Bước 2: gom ảnh theo train/val/test, tự chia nếu chưa có split.
    records_by_split = _group_by_split(records, config)

    datasets: dict[str, ImageDataset] = {}
    for split_name, split_records_ in records_by_split.items():
        # Train dùng augmentation, val/test dùng transform cố định.
        transform = (
            build_train_transform(config.image_size)
            if split_name == "train"
            else build_eval_transform(config.image_size)
        )
        datasets[split_name] = ImageDataset(
            records=split_records_,
            transform=transform,
            return_metadata=config.return_metadata,
        )

    return datasets


def build_dataloaders(config: DataLoaderConfig) -> dict[str, object]:
    """Tạo PyTorch DataLoader cho từng split."""

    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Cài PyTorch để tạo DataLoader: pip install torch") from exc

    loaders: dict[str, object] = {}
    for split_name, dataset in build_datasets(config).items():
        # Train cần shuffle, val/test giữ thứ tự cố định để đánh giá ổn định.
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split_name == "train"),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=(split_name == "train" and config.drop_last_train_batch),
        )
    return loaders


def _load_records(config: DataLoaderConfig) -> list[ImageRecord]:
    # Dataset được đọc trực tiếp từ cấu trúc folder.
    return discover_image_records(config.data_root)


def _group_by_split(
    records: Sequence[ImageRecord], config: DataLoaderConfig
) -> dict[str, list[ImageRecord]]:
    # Trường hợp dataset đã có split đầy đủ.
    if all(record.split for record in records):
        records = list(records)
    # Trường hợp dataset chưa có split, tự chia theo tỷ lệ trong config.
    elif not any(record.split for record in records):
        records = split_records(
            records,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
    else:
        raise ValueError("Dataset phải có split cho tất cả ảnh hoặc không ảnh nào có split.")

    result: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        # Chỉ chấp nhận 3 split chính để tránh lỗi train sai tập.
        if record.split not in result:
            raise ValueError(f"Split không hợp lệ ở ảnh {record.path}: {record.split}")
        result[record.split].append(record)

    return {split: items for split, items in result.items() if items}
