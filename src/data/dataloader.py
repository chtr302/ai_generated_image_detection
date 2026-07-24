"""Build PyTorch Dataset and DataLoader from local folders or Hugging Face."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .dataset import (
    HuggingFaceImageDataset,
    ImageDataset,
    ImageRecord,
    StreamingHuggingFaceImageDataset,
    discover_image_records,
    split_records,
)
from .transforms import build_eval_transform, build_train_transform


DEFAULT_HF_DATASET = "Rajarshi-Roy-research/Defactify_Image_Dataset"


@dataclass(frozen=True)
class DataLoaderConfig:
    # Use data_root for local folders, or hf_dataset_id for Hugging Face.
    data_root: str | Path | None = None
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

    # Hugging Face Defactify defaults.
    hf_dataset_id: str | None = None
    hf_config_name: str | None = None
    hf_cache_dir: str | Path | None = None
    hf_train_split: str = "train"
    hf_val_split: str = "validation"
    hf_test_split: str = "test"
    hf_image_column: str = "Image"
    hf_label_column: str = "Label_A"
    hf_caption_column: str = "Caption"
    hf_source_label_column: str = "Label_B"
    hf_trust_remote_code: bool = False
    hf_streaming: bool = True
    hf_shuffle_buffer: int = 10_000


def build_datasets(config: DataLoaderConfig) -> dict[str, ImageDataset | HuggingFaceImageDataset | StreamingHuggingFaceImageDataset]:
    """Build datasets for train, val, and test splits."""

    if config.hf_dataset_id:
        return _build_hf_datasets(config)
    return _build_folder_datasets(config)


def build_dataloaders(config: DataLoaderConfig) -> dict[str, object]:
    """Build PyTorch DataLoader for each split."""

    try:
        from torch.utils.data import DataLoader, IterableDataset
    except ImportError as exc:
        raise ImportError("Install PyTorch to build DataLoader: pip install torch") from exc

    loaders: dict[str, object] = {}
    for split_name, dataset in build_datasets(config).items():
        is_iterable = isinstance(dataset, IterableDataset)
        loader_kwargs = {
            "batch_size": config.batch_size,
            "shuffle": split_name == "train" and not is_iterable,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "drop_last": split_name == "train" and config.drop_last_train_batch,
        }
        if config.num_workers > 0 and not is_iterable:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        loaders[split_name] = DataLoader(dataset, **loader_kwargs)
    return loaders


def _build_folder_datasets(config: DataLoaderConfig) -> dict[str, ImageDataset]:
    if config.data_root is None:
        raise ValueError("Pass data_root for local folders or hf_dataset_id for Hugging Face.")

    records = discover_image_records(config.data_root)
    records_by_split = _group_by_split(records, config)

    datasets: dict[str, ImageDataset] = {}
    for split_name, split_records_ in records_by_split.items():
        transform = build_train_transform(config.image_size) if split_name == "train" else build_eval_transform(config.image_size)
        datasets[split_name] = ImageDataset(
            records=split_records_,
            transform=transform,
            return_metadata=config.return_metadata,
        )
    return datasets


def _build_hf_datasets(config: DataLoaderConfig) -> dict[str, HuggingFaceImageDataset | StreamingHuggingFaceImageDataset]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install Hugging Face datasets to read Defactify: pip install datasets") from exc

    split_map = {
        "train": config.hf_train_split,
        "val": config.hf_val_split,
        "test": config.hf_test_split,
    }
    datasets: dict[str, HuggingFaceImageDataset | StreamingHuggingFaceImageDataset] = {}
    for local_split, hf_split in split_map.items():
        if not hf_split:
            continue

        kwargs = {
            "split": hf_split,
            "cache_dir": str(config.hf_cache_dir) if config.hf_cache_dir else None,
            "trust_remote_code": config.hf_trust_remote_code,
            "streaming": config.hf_streaming,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if config.hf_config_name:
            raw_split = load_dataset(config.hf_dataset_id, config.hf_config_name, **kwargs)
        else:
            raw_split = load_dataset(config.hf_dataset_id, **kwargs)

        if config.hf_streaming:
            raw_split = _prepare_streaming_split(raw_split, local_split, config)

        transform = build_train_transform(config.image_size) if local_split == "train" else build_eval_transform(config.image_size)
        dataset_cls = StreamingHuggingFaceImageDataset if config.hf_streaming else HuggingFaceImageDataset
        datasets[local_split] = dataset_cls(
            dataset=raw_split,
            image_column=config.hf_image_column,
            label_column=config.hf_label_column,
            caption_column=config.hf_caption_column,
            source_label_column=config.hf_source_label_column,
            transform=transform,
            return_metadata=config.return_metadata,
        )
    return datasets


def _prepare_streaming_split(dataset, local_split: str, config: DataLoaderConfig):
    if local_split == "train" and config.hf_shuffle_buffer > 0 and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(buffer_size=config.hf_shuffle_buffer, seed=config.seed)

    try:
        import torch.distributed as dist
        from datasets.distributed import split_dataset_by_node
    except ImportError:
        return dataset

    if dist.is_available() and dist.is_initialized():
        dataset = split_dataset_by_node(dataset, rank=dist.get_rank(), world_size=dist.get_world_size())
    return dataset


def _group_by_split(records: Sequence[ImageRecord], config: DataLoaderConfig) -> dict[str, list[ImageRecord]]:
    if all(record.split for record in records):
        records = list(records)
    elif not any(record.split for record in records):
        records = split_records(
            records,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
    else:
        raise ValueError("Dataset must define splits for all images, or for none of them.")

    result: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    for record in records:
        if record.split not in result:
            raise ValueError(f"Invalid split at image {record.path}: {record.split}")
        result[record.split].append(record)

    return {split: items for split, items in result.items() if items}
