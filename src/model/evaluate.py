from __future__ import annotations

import torch
import torch.distributed as dist

from src.model.batch import unpack_batch


@torch.no_grad()
def evaluate_binary_classifier(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    max_steps: int | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(6, device=device)
    processed_samples = 0

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        if max_samples is not None and processed_samples >= max_samples:
            break

        images, labels = unpack_batch(batch, device)
        if max_samples is not None:
            remaining = max_samples - processed_samples
            if remaining <= 0:
                break
            if labels.numel() > remaining:
                images = images[:remaining]
                labels = labels[:remaining]
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            outputs = model(images)

        probs = outputs["prob_ai"]
        preds = (probs >= 0.5).float()
        totals[0] += (preds == labels).sum()
        totals[1] += labels.numel()
        totals[2] += ((preds == 1) & (labels == 1)).sum()
        totals[3] += (preds == 1).sum()
        totals[4] += ((preds == 1) & (labels == 0)).sum()
        totals[5] += ((preds == 0) & (labels == 1)).sum()
        processed_samples += labels.numel()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    correct, count, true_positive, predicted_positive, false_positive, false_negative = totals.tolist()
    actual_positive = true_positive + false_negative
    actual_negative = count - actual_positive
    true_negative = actual_negative - false_positive

    accuracy = correct / max(count, 1.0)
    precision = true_positive / max(predicted_positive, 1.0)
    recall = true_positive / max(actual_positive, 1.0)
    specificity = true_negative / max(actual_negative, 1.0)
    balanced_accuracy = 0.5 * (recall + specificity)
    predicted_ai_rate = predicted_positive / max(count, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "specificity_real": specificity,
        "f1_ai": f1,
        "predicted_ai_rate": predicted_ai_rate,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "sample_count": count,
    }
