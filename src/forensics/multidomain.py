from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.forensics.fft_extractor import FFTExtractor
from src.forensics.dwt_extractor import DWTExtractor


class MultiDomainPreprocessor(nn.Module):
    """Tao ba bieu dien: spatial, frequency va wavelet."""

    def __init__(self) -> None:
        super().__init__()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3))
        self.fft = FFTExtractor()
        self.dwt = DWTExtractor()

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "spatial": self._spatial(image),
            "frequency": self.fft(image),
            "wavelet": self.dwt(image),
        }

    def _spatial(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        kernel = self.laplacian.repeat(channels, 1, 1, 1)
        edges = F.conv2d(image, kernel, padding=1, groups=channels)
        return torch.cat([image, edges], dim=1)
