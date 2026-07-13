from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.data.dataloader import DataLoaderConfig, build_datasets
from src.model.architectures.detector import build_detector
from src.model.batch import unpack_batch
from src.model.evaluate import evaluate_binary_classifier
from src.model.loss import DetectionLoss, sparsity_for_nec
from src.xai.concept_bottleneck import DEFAULT_CONCEPTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI-generated image detector with optional DDP.")
    parser.add_argument("--data-root", default=None, help="Root co train/val/test hoac real/ai.")
    parser.add_argument("--train-dir", default=None, help="Legacy alias, vi du data/train.")
    parser.add_argument("--val-dir", default=None, help="Legacy only; val duoc doc tu data-root neu co.")
    parser.add_argument("--output-dir", default="outputs/ai_detector")
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
        )
        val_metrics = evaluate_binary_classifier(model, val_loader, device, amp_dtype) if val_loader else {}

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
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, device=device)

    for step, batch in enumerate(loader):
        images, labels = unpack_batch(batch, device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            outputs = model(images)
            loss, metrics = criterion(outputs, labels)
            scaled_loss = loss / grad_accum

        scaler.scale(scaled_loss).backward()
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        preds = (outputs["prob_ai"].detach() >= 0.5).float()
        totals[0] += metrics["loss"] * labels.numel()
        totals[1] += (preds == labels).sum()
        totals[2] += labels.numel()

        if is_main and step % 25 == 0:
            print(f"epoch={epoch} step={step}/{len(loader)} loss={float(metrics['loss']):.4f}")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss, correct, count = totals.tolist()
    return {"loss": total_loss / max(count, 1.0), "accuracy": correct / max(count, 1.0)}


def build_train_loaders(
    data_root: str | Path,
    batch_size: int,
    image_size: int,
    num_workers: int,
    distributed: bool,
    pin_memory: bool,
) -> tuple[dict[str, DataLoader], DistributedSampler | None]:
    config = DataLoaderConfig(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    datasets = build_datasets(config)
    if "train" not in datasets:
        raise ValueError("Dataset can co split train hoac anh chua chia split de tu tach train/val/test.")

    train_sampler: DistributedSampler | None = None
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        sampler = None
        shuffle = split == "train"
        drop_last = split == "train"
        if split == "train" and distributed:
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


def _resolve_data_root(args: argparse.Namespace) -> Path:
    if args.data_root:
        return Path(args.data_root)
    if args.train_dir:
        train_dir = Path(args.train_dir)
        return train_dir.parent if train_dir.name.lower() in {"train", "training"} else train_dir
    raise ValueError("Can truyen --data-root, hoac --train-dir de tu suy ra data root.")


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

