from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.forensics.multidomain import MultiDomainPreprocessor
from src.model.architectures.branch_encoder import BranchEncoder
from src.model.architectures.fusion import CrossDomainAttentionFusion
from src.model.architectures.swin_blocks import SwinBackbone
from src.xai.concept_bottleneck import ConceptBottleneckHead


@dataclass(frozen=True)
class DetectorConfig:
    image_size: int = 384
    branch_channels: int = 256
    concept_count: int = 15
    nec: int = 10
    dropout: float = 0.1


class ExplainableAIGeneratedImageDetector(nn.Module):
    """SFW-SwinCBM: Spatial, Frequency, Wavelet, Swin va CBM."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DetectorConfig()
        channels = self.config.branch_channels
        self.preprocessor = MultiDomainPreprocessor()
        self.branches = nn.ModuleDict(
            {
                "spatial": BranchEncoder(6, channels),
                "frequency": BranchEncoder(3, channels),
                "wavelet": BranchEncoder(9, channels),
            }
        )
        self.fusion = CrossDomainAttentionFusion(channels=channels, heads=8, branch_count=3)
        self.backbone = SwinBackbone(in_channels=channels, dropout=self.config.dropout)
        self.head = ConceptBottleneckHead(
            in_features=1024,
            concept_count=self.config.concept_count,
            nec=self.config.nec,
            dropout=self.config.dropout,
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        domains = self.preprocessor(image)
        branch_features = {name: self.branches[name](tensor) for name, tensor in domains.items()}
        fused = self.fusion(branch_features)
        z, feature_maps = self.backbone(fused)
        output = self.head(z)
        output["z"] = z
        output["fused"] = fused
        output["heatmap"] = _simple_heatmap(fused, image_size=image.shape[-2:])
        output["features"] = {**branch_features, **feature_maps}
        return output


def build_detector(nec: int = 10, concept_count: int = 15, image_size: int = 384) -> ExplainableAIGeneratedImageDetector:
    config = DetectorConfig(image_size=image_size, nec=nec, concept_count=concept_count)
    return ExplainableAIGeneratedImageDetector(config)


def _simple_heatmap(features: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    heatmap = features.abs().mean(dim=1, keepdim=True)
    heatmap = torch.nn.functional.interpolate(heatmap, size=image_size, mode="bilinear", align_corners=False)
    flat = heatmap.flatten(1)
    min_value = flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_value = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (heatmap - min_value) / (max_value - min_value).clamp_min(1e-6)
