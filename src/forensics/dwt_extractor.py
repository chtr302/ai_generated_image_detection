from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTExtractor(nn.Module):
    """DWT db4 detail bands LH, HL, HH cho texture vi mo."""

    def __init__(self) -> None:
        super().__init__()
        low = torch.tensor(
            [
                -0.010597401785,
                0.032883011667,
                0.030841381836,
                -0.187034811719,
                -0.027983769417,
                0.630880767930,
                0.714846570553,
                0.230377813309,
            ],
            dtype=torch.float32,
        )
        high = torch.flip(low, dims=(0,))
        high[::2] *= -1.0
        filters = torch.stack(
            [
                torch.outer(low, high),
                torch.outer(high, low),
                torch.outer(high, high),
            ]
        )
        self.register_buffer("filters", filters[:, None])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        filters = self.filters.repeat(channels, 1, 1, 1)
        image = F.pad(image, (3, 4, 3, 4), mode="reflect")
        details = F.conv2d(image, filters, stride=2, groups=channels)
        details = details.view(batch, channels * 3, details.shape[-2], details.shape[-1])
        return F.interpolate(details, size=(height, width), mode="bilinear", align_corners=False)
