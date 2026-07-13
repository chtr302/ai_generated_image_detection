from __future__ import annotations

import math

import torch
import torch.nn as nn


class DCTExtractor(nn.Module):
    """DCT 2D de nhan manh artifact trung/cao tan."""

    def __init__(self, low_frequency_ratio: float = 0.18) -> None:
        super().__init__()
        self.low_frequency_ratio = low_frequency_ratio

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        dct_h = _dct_matrix(height, image.device, image.dtype)
        dct_w = _dct_matrix(width, image.device, image.dtype)
        coeffs = torch.matmul(torch.matmul(dct_h, image), dct_w.transpose(0, 1))
        coeffs = torch.log1p(coeffs.abs())
        coeffs = self._suppress_low_frequency(coeffs)
        mean = coeffs.mean(dim=(-2, -1), keepdim=True)
        std = coeffs.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (coeffs - mean) / std

    def _suppress_low_frequency(self, coeffs: torch.Tensor) -> torch.Tensor:
        height, width = coeffs.shape[-2:]
        cut_h = max(1, int(height * self.low_frequency_ratio))
        cut_w = max(1, int(width * self.low_frequency_ratio))
        mask = torch.ones((height, width), device=coeffs.device, dtype=coeffs.dtype)
        mask[:cut_h, :cut_w] = 0.15
        return coeffs * mask


def _dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(size, device=device, dtype=dtype)
    k = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    matrix = torch.cos(math.pi / size * (n + 0.5) * k)
    matrix[0] *= (1.0 / size) ** 0.5
    matrix[1:] *= (2.0 / size) ** 0.5
    return matrix
