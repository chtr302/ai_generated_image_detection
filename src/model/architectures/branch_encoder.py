from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class MBConv(nn.Module):
    """Mobile inverted block nhe cho forensic branch."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand_ratio: int = 4) -> None:
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden, kernel_size=1),
            ConvBNAct(hidden, hidden, kernel_size=3, stride=stride, groups=hidden),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return F.silu(out)


class BranchEncoder(nn.Module):
    """Encoder rieng cho spatial/frequency/wavelet/camera branch."""

    def __init__(self, in_channels: int, out_channels: int = 256, base_channels: int = 48) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNAct(in_channels, base_channels, stride=2),
            MBConv(base_channels, base_channels * 2, stride=2),
            MBConv(base_channels * 2, base_channels * 4, stride=2),
            MBConv(base_channels * 4, base_channels * 6, stride=2),
            MBConv(base_channels * 6, base_channels * 8, stride=2),
            nn.Conv2d(base_channels * 8, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
