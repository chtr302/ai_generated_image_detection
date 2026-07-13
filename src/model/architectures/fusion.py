from __future__ import annotations

import torch
import torch.nn as nn


class CrossDomainAttentionFusion(nn.Module):
    """Dung spatial lam query, frequency/wavelet lam evidence."""

    def __init__(self, channels: int = 256, heads: int = 8, branch_count: int = 3) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * branch_count, channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, branch_count, kernel_size=1),
            nn.Softmax(dim=1),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, branches: dict[str, torch.Tensor]) -> torch.Tensor:
        spatial = branches["spatial"]
        ordered = [branches["spatial"], branches["frequency"], branches["wavelet"]]
        batch, channels, height, width = spatial.shape

        query = spatial.flatten(2).transpose(1, 2)
        evidence = torch.cat(ordered[1:], dim=2).flatten(2).transpose(1, 2)
        attended, _ = self.attn(query, evidence, evidence, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)

        gates = self.gate(torch.cat(ordered, dim=1))
        gated = sum(gates[:, index : index + 1] * value for index, value in enumerate(ordered))
        return self.proj(torch.cat([gated, attended], dim=1))
