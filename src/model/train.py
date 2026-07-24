from __future__ import annotations

import argparse
import json
import os
import time
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
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size.")
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
    parser.add_argument("--log-every", type=int, default=100, help="Update progress metrics every N steps.")
    parser.add_argument("--progress", choices=["bar", "text", "none"], default="bar", help="Progress display style.")
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

    if is_main:
        print(_format_train_setup(args, device, ddp, train_loader))

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
            total_epochs=args.epochs,
            progress=args.progress,
        )
        val_metrics = evaluate_binary_classifier(model, val_loader, device, amp_dtype, args.max_val_steps) if _loader_exists(val_loader) else {}

        if is_main:
            print(_format_epoch_summary(epoch, train_metrics, val_metrics))
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
    total_epochs: int | None = None,
    progress: str = "bar",
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(3, device=device)
    pending_steps = 0
    planned_steps = _planned_steps(loader, max_steps)
    start_time = time.perf_counter()
    progress_bar = _build_progress_bar(
        enabled=is_main and progress == "bar",
        total=planned_steps,
        description=_epoch_label(epoch, total_epochs),
    )

    if progress_bar is None and is_main and progress != "none":
        print(_format_epoch_start(epoch, total_epochs, planned_steps))

    try:
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

            current_step = step + 1
            should_report = current_step == 1 or (log_every > 0 and current_step % log_every == 0)

            if progress_bar is not None:
                progress_bar.update(1)
                if should_report:
                    sample_count = max(float(totals[2].item()), 1.0)
                    elapsed = time.perf_counter() - start_time
                    running_loss = float(totals[0].item() / sample_count)
                    running_acc = float(totals[1].item() / sample_count)
                    samples_per_sec = sample_count / max(elapsed, 1e-6)
                    progress_bar.set_postfix(
                        loss=f"{running_loss:.4f}",
                        acc=f"{running_acc:.4f}",
                        img_s=f"{samples_per_sec:.1f}",
                    )
            elif is_main and progress != "none" and should_report:
                sample_count = max(float(totals[2].item()), 1.0)
                elapsed = time.perf_counter() - start_time
                running_loss = float(totals[0].item() / sample_count)
                running_acc = float(totals[1].item() / sample_count)
                samples_per_sec = sample_count / max(elapsed, 1e-6)
                print(
                    _format_train_progress(
                        step=current_step,
                        total_steps=planned_steps,
                        batch_loss=float(metrics["loss"]),
                        running_loss=running_loss,
                        running_acc=running_acc,
                        sample_count=int(sample_count),
                        samples_per_sec=samples_per_sec,
                        elapsed=elapsed,
                    )
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if pending_steps > 0:
        _optimizer_step(model, optimizer, scaler)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss, correct, count = totals.tolist()
    elapsed = time.perf_counter() - start_time
    return {
        "loss": total_loss / max(count, 1.0),
        "accuracy": correct / max(count, 1.0),
        "samples": count,
        "seconds": elapsed,
        "samples_per_sec": count / max(elapsed, 1e-6),
    }


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

        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }
        if num_workers > 0 and not is_iterable:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        loaders[split] = DataLoader(dataset, **loader_kwargs)
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


def _planned_steps(loader, max_steps: int | None) -> int | None:
    loader_len = _safe_len(loader)
    if max_steps is None:
        return loader_len
    if loader_len is None:
        return max_steps
    return min(loader_len, max_steps)


def _epoch_label(epoch: int, total_epochs: int | None) -> str:
    return f"Epoch {epoch + 1}/{total_epochs}" if total_epochs is not None else f"Epoch {epoch + 1}"


def _build_progress_bar(enabled: bool, total: int | None, description: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=description, unit="step", dynamic_ncols=True, leave=True)


def _format_train_setup(args: argparse.Namespace, device: torch.device, ddp: bool, train_loader) -> str:
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    effective_batch = args.batch_size * args.grad_accum * world_size
    planned_steps = _planned_steps(train_loader, args.max_train_steps)
    train_steps = str(planned_steps) if planned_steps is not None else "full split"
    lines = [
        "",
        "Training setup",
        "=" * 88,
        f"device={device} | ddp={ddp} | amp={args.amp} | workers={args.num_workers}",
        (
            f"image_size={args.image_size} | batch/gpu={args.batch_size} | "
            f"grad_accum={args.grad_accum} | effective_batch={effective_batch}"
        ),
        f"epochs={args.epochs} | train_steps/epoch={train_steps} | val_steps={args.max_val_steps or 'full'}",
        "=" * 88,
    ]
    return "\n".join(lines)


def _format_epoch_start(epoch: int, total_epochs: int | None, planned_steps: int | None) -> str:
    epoch_label = f"{epoch + 1}/{total_epochs}" if total_epochs is not None else str(epoch + 1)
    step_label = str(planned_steps) if planned_steps is not None else "?"
    return "\n".join(
        [
            "",
            f"Epoch {epoch_label} | train_steps={step_label}",
            "-" * 88,
            f"{'step':>10} {'batch_loss':>11} {'avg_loss':>10} {'avg_acc':>9} {'samples':>9} {'img/s':>9} {'elapsed':>9} {'eta':>9}",
        ]
    )


def _format_train_progress(
    step: int,
    total_steps: int | None,
    batch_loss: float,
    running_loss: float,
    running_acc: float,
    sample_count: int,
    samples_per_sec: float,
    elapsed: float,
) -> str:
    if total_steps is None:
        step_text = f"{step}/?"
        eta_text = "-"
    else:
        step_text = f"{step}/{total_steps}"
        seconds_per_step = elapsed / max(step, 1)
        eta_text = _format_duration(max(total_steps - step, 0) * seconds_per_step)
    return (
        f"{step_text:>10} "
        f"{batch_loss:>11.4f} "
        f"{running_loss:>10.4f} "
        f"{running_acc:>9.4f} "
        f"{sample_count:>9} "
        f"{samples_per_sec:>9.1f} "
        f"{_format_duration(elapsed):>9} "
        f"{eta_text:>9}"
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_epoch_summary(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> str:
    samples = int(train_metrics.get("samples", 0.0))
    elapsed = _format_duration(train_metrics.get("seconds"))
    raw_speed = train_metrics.get("samples_per_sec")
    speed = "-" if raw_speed is None else f"{raw_speed:.1f}"
    lines = [
        "",
        f"Epoch {epoch + 1} summary | samples={samples} | time={elapsed} | speed={speed} img/s",
        "-" * 76,
        f"{'split':<8} {'loss':>9} {'acc':>9} {'precision':>10} {'recall':>9} {'f1_ai':>9} {'bal_acc':>9}",
        _format_metric_row("train", train_metrics),
    ]
    if val_metrics:
        lines.append(_format_metric_row("val", val_metrics))
    lines.append("-" * 76)
    return "\n".join(lines)


def _format_metric_row(split: str, metrics: dict[str, float]) -> str:
    return (
        f"{split:<8} "
        f"{_format_metric(metrics.get('loss')):>9} "
        f"{_format_metric(metrics.get('accuracy')):>9} "
        f"{_format_metric(metrics.get('precision_ai')):>10} "
        f"{_format_metric(metrics.get('recall_ai')):>9} "
        f"{_format_metric(metrics.get('f1_ai')):>9} "
        f"{_format_metric(metrics.get('balanced_accuracy')):>9}"
    )


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


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
