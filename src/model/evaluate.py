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
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(6, device=device)

    for batch in loader:
        images, labels = unpack_batch(batch, device)
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

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    correct, count, true_positive, predicted_positive, false_positive, false_negative = totals.tolist()
    accuracy = correct / max(count, 1.0)
    precision = true_positive / max(predicted_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
