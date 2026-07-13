from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WindowAttentionBlock(nn.Module):
    """Swin-style block voi window attention va optional shift."""

    def __init__(self, dim: int, heads: int, window_size: int, shift_size: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        _, channels, height, width = x.shape
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        windows, padding = window_partition(x, self.window_size)
        tokens = windows.flatten(2).transpose(1, 2)
        norm_tokens = self.norm1(tokens)
        tokens = tokens + self.attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)[0]
        tokens = tokens + self.mlp(self.norm2(tokens))
        windows = tokens.transpose(1, 2).reshape(-1, channels, self.window_size, self.window_size)
        x = window_reverse(windows, self.window_size, height, width, padding)

        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return x + residual


class PatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduction(x)


class SwinBackbone(nn.Module):
    """Backbone hoc quan he giua cac artifact sau fusion."""

    def __init__(self, in_channels: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            WindowAttentionBlock(in_channels, heads=8, window_size=6, shift_size=0, dropout=dropout),
            WindowAttentionBlock(in_channels, heads=8, window_size=6, shift_size=3, dropout=dropout),
        )
        self.merge1 = PatchMerging(in_channels, 512)
        self.stage2 = nn.Sequential(
            WindowAttentionBlock(512, heads=8, window_size=3, shift_size=0, dropout=dropout),
            WindowAttentionBlock(512, heads=8, window_size=3, shift_size=1, dropout=dropout),
        )
        self.merge2 = PatchMerging(512, 1024)
        self.stage3 = WindowAttentionBlock(1024, heads=8, window_size=3, shift_size=0, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stage1 = self.stage1(x)
        stage2 = self.stage2(self.merge1(stage1))
        stage3 = self.stage3(self.merge2(stage2))
        z = self.pool(stage3).flatten(1)
        return z, {"swin_stage1": stage1, "swin_stage2": stage2, "swin_stage3": stage3}


def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    batch, channels, height, width = x.shape
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    _, _, padded_h, padded_w = x.shape
    x = x.view(batch, channels, padded_h // window_size, window_size, padded_w // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, channels, window_size, window_size)
    return windows, (pad_h, pad_w)


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
    padding: tuple[int, int],
) -> torch.Tensor:
    pad_h, pad_w = padding
    padded_h = height + pad_h
    padded_w = width + pad_w
    batch = windows.shape[0] // ((padded_h // window_size) * (padded_w // window_size))
    channels = windows.shape[1]
    x = windows.view(batch, padded_h // window_size, padded_w // window_size, channels, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, padded_h, padded_w)
    return x[:, :, :height, :width]
