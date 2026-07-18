from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataloader import DEFAULT_HF_DATASET, DataLoaderConfig, build_datasets
from src.model.architectures.detector import build_detector
from src.model.batch import unpack_batch
from src.model.evaluate import evaluate_binary_classifier
from src.model.loss import DetectionLoss, sparsity_for_nec
from src.xai.concept_bottleneck import DEFAULT_CONCEPTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI-generated image detector with optional DDP.")
    parser.add_argument("--data-root", default=None, help="Root folder with train/val/test or real/ai folders.")
    parser.add_argument("--train-dir", default=None, help="Legacy alias, for example data/train.")
    parser.add_argument("--val-dir", default=None, help="Legacy only; val is read from data-root when available.")
    parser.add_argument("--output-dir", default="outputs/ai_detector")
    parser.add_argument("--hf-dataset", default=None, help=f"Hugging Face dataset id, for example {DEFAULT_HF_DATASET}.")
    parser.add_argument("--hf-config-name", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--hf-trust-remote-code", action="store_true")
    parser.add_argument("--hf-no-streaming", action="store_true", help="Disable Hugging Face streaming and cache the split locally.")
    parser.add_argument("--hf-shuffle-buffer", type=int, default=10_000, help="Streaming shuffle buffer for train split.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--nec", type=int, default=10)
    parser.add_argument("--concept-count", type=int, default=len(DEFAULT_CONCEPTS))
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-train-steps", type=int, default=None, help="Limit train batches per epoch for streaming/debug runs.")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Limit validation batches per epoch for faster Colab runs.")
    parser.add_argument("--log-every", type=int, default=100, help="Print train loss every N steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if ddp:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = not ddp or dist.get_rank() == 0
    amp_dtype = _amp_dtype(args.amp)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "concepts.json").write_text(json.dumps(DEFAULT_CONCEPTS[: args.concept_count], indent=2))

    model = build_detector(nec=args.nec, concept_count=args.concept_count, image_size=args.image_size).to(device)
    if ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    criterion = DetectionLoss(sparsity_weight=sparsity_for_nec(args.nec)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = build_grad_scaler(enabled=args.amp == "fp16" and device.type == "cuda")
    start_epoch = _load_checkpoint(args.resume, model, optimizer, device)

    data_root = _resolve_data_root(args)
    loaders, train_sampler = build_train_loaders(
        data_root=data_root,
        hf_dataset_id=args.hf_dataset,
        hf_config_name=args.hf_config_name,
        hf_cache_dir=args.hf_cache_dir,
        hf_trust_remote_code=args.hf_trust_remote_code,
        hf_streaming=not args.hf_no_streaming,
        hf_shuffle_buffer=args.hf_shuffle_buffer,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        distributed=ddp,
        pin_memory=device.type == "cuda",
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    best_score = 0.0
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        else:
            train_dataset = getattr(train_loader, "dataset", None)
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            grad_accum=args.grad_accum,
            epoch=epoch,
            is_main=is_main,
            max_steps=args.max_train_steps,
            log_every=args.log_every,
        )
        val_metrics = evaluate_binary_classifier(model, val_loader, device, amp_dtype, args.max_val_steps) if _loader_exists(val_loader) else {}

        if is_main:
            print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}, indent=2))
            score = val_metrics.get("accuracy", train_metrics["accuracy"])
            _save_checkpoint(output_dir / "last.pt", epoch, model, optimizer, args)
            if score >= best_score:
                best_score = score
                _save_checkpoint(output_dir / "best.pt", epoch, model, optimizer, args)

    if ddp:
        dist.destroy_process_group()


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: DetectionLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    grad_accum: int,
    epoch: int,
    is_main: bool,
    max_steps: int | None = None,
    log_every: int = 100,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, device=device)
    pending_steps = 0
    loader_len = _safe_len(loader)

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        images, labels = unpack_batch(batch, device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            outputs = model(images)
            loss, metrics = criterion(outputs, labels)
            scaled_loss = loss / grad_accum

        scaler.scale(scaled_loss).backward()
        pending_steps += 1
        if pending_steps >= grad_accum:
            _optimizer_step(model, optimizer, scaler)
            pending_steps = 0

        preds = (outputs["prob_ai"].detach() >= 0.5).float()
        totals[0] += metrics["loss"] * labels.numel()
        totals[1] += (preds == labels).sum()
        totals[2] += labels.numel()

        if is_main and log_every > 0 and step % log_every == 0:
            total_steps = str(loader_len) if loader_len is not None else "?"
            print(f"epoch={epoch} step={step}/{total_steps} loss={float(metrics['loss']):.4f}")

    if pending_steps > 0:
        _optimizer_step(model, optimizer, scaler)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss, correct, count = totals.tolist()
    return {"loss": total_loss / max(count, 1.0), "accuracy": correct / max(count, 1.0)}


def build_train_loaders(
    data_root: str | Path | None,
    batch_size: int,
    image_size: int,
    num_workers: int,
    distributed: bool,
    pin_memory: bool,
    hf_dataset_id: str | None = None,
    hf_config_name: str | None = None,
    hf_cache_dir: str | Path | None = None,
    hf_trust_remote_code: bool = False,
    hf_streaming: bool = True,
    hf_shuffle_buffer: int = 10_000,
) -> tuple[dict[str, DataLoader], DistributedSampler | None]:
    config = DataLoaderConfig(
        data_root=data_root,
        hf_dataset_id=hf_dataset_id,
        hf_config_name=hf_config_name,
        hf_cache_dir=hf_cache_dir,
        hf_trust_remote_code=hf_trust_remote_code,
        hf_streaming=hf_streaming,
        hf_shuffle_buffer=hf_shuffle_buffer,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    datasets = build_datasets(config)
    if "train" not in datasets:
        raise ValueError("Dataset needs a train split or unsplit images that can be split automatically.")

    train_sampler: DistributedSampler | None = None
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        sampler = None
        is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
        shuffle = split == "train" and not is_iterable
        drop_last = False
        if split == "train" and distributed and not is_iterable:
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=drop_last)
            train_sampler = sampler
            shuffle = False

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    return loaders, train_sampler


def _loader_exists(loader) -> bool:
    return loader is not None


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _safe_len(loader) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def _resolve_data_root(args: argparse.Namespace) -> Path | None:
    if args.hf_dataset:
        return Path(args.data_root) if args.data_root else None
    if args.data_root:
        return Path(args.data_root)
    if args.train_dir:
        train_dir = Path(args.train_dir)
        return train_dir.parent if train_dir.name.lower() in {"train", "training"} else train_dir
    raise ValueError("Pass --data-root for local folders, or --hf-dataset for Hugging Face.")


def _load_checkpoint(
    checkpoint_path: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not checkpoint_path:
        return 0
    checkpoint = torch.load(checkpoint_path, map_location=device)
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", -1)) + 1


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    target_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "model": target_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def build_grad_scaler(enabled: bool):
    grad_scaler = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if grad_scaler is not None:
        try:
            return grad_scaler("cuda", enabled=enabled)
        except TypeError:
            return grad_scaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _amp_dtype(amp: str) -> torch.dtype | None:
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        return torch.bfloat16
    return None


if __name__ == "__main__":
    main()
