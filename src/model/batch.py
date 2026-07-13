from __future__ import annotations

from typing import Any

import torch


def unpack_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Lay image/label tu batch dang dict hoac tuple."""

    if isinstance(batch, dict):
        images = batch["image"]
        labels = batch["label"]
    elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
        images, labels = batch[0], batch[1]
    else:
        raise TypeError("Batch phai la dict hoac tuple/list co image va label.")

    labels = labels.float() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.float32)
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
