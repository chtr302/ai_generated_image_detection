from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemosaicingExtractor(nn.Module):
    """Residual sau low-pass de bat dau vet noi suy camera/generator."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        smooth = F.avg_pool2d(image, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        residual = image - smooth
        return residual
