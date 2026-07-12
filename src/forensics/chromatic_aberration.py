from __future__ import annotations

import torch
import torch.nn as nn


class ChromaticAberrationExtractor(nn.Module):
    """Sai khac giua cac kenh mau sau khi tru mean khong gian."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        red = image[:, 0:1]
        green = image[:, 1:2]
        blue = image[:, 2:3]
        rg = red - green
        gb = green - blue
        rb = red - blue
        residual = torch.cat([rg, gb, rb], dim=1)
        residual = residual - residual.mean(dim=(-2, -1), keepdim=True)
        return residual
