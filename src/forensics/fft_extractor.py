from __future__ import annotations

import torch
import torch.nn as nn


class FFTExtractor(nn.Module):
    """Trich xuat dau vet mien tan so bang FFT."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft2(image, norm="ortho")
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(spectrum))
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (magnitude - mean) / std
