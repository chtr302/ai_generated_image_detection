from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_CONCEPTS = [
    "over_smooth_texture",
    "checkerboard_artifact",
    "high_frequency_spike",
    "wavelet_noise_inconsistency",
    "blending_boundary",
    "geometry_asymmetry",
    "shadow_reflection_error",
    "background_pattern_repetition",
    "skin_hair_detail_mismatch",
    "compression_residual_mismatch",
    "local_texture_discontinuity",
    "unnatural_edge_transition",
    "semantic_detail_conflict",
    "upsampling_trace",
    "color_channel_inconsistency",
]


class ConceptBottleneckHead(nn.Module):
    """Bien vector dac trung thanh concepts roi moi phan loai."""

    def __init__(self, in_features: int = 1024, concept_count: int = 15, nec: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.concept_count = concept_count
        self.nec = min(nec, concept_count)
        self.mapper = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, concept_count),
        )
        self.classifier = nn.Linear(concept_count, 1)
        self.uncertainty = nn.Sequential(
            nn.Linear(in_features + concept_count, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        concept_logits = self.mapper(z)
        concepts = torch.sigmoid(concept_logits)
        effective_concepts, mask = self._select_effective_concepts(concepts)
        logits = self.classifier(effective_concepts).squeeze(1)
        uncertainty_logits = self.uncertainty(torch.cat([z, effective_concepts], dim=1)).squeeze(1)
        return {
            "logits": logits,
            "prob_ai": torch.sigmoid(logits),
            "concept_logits": concept_logits,
            "concepts": concepts,
            "effective_concepts": effective_concepts,
            "concept_mask": mask,
            "uncertainty_logits": uncertainty_logits,
            "uncertainty": torch.sigmoid(uncertainty_logits),
        }

    def _select_effective_concepts(self, concepts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.nec >= self.concept_count:
            mask = torch.ones_like(concepts)
            return concepts, mask
        indices = torch.topk(concepts, k=self.nec, dim=1).indices
        mask = torch.zeros_like(concepts)
        mask.scatter_(1, indices, 1.0)
        return concepts * mask, mask
