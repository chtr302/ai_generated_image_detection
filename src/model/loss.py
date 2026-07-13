from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """Joint loss: classification, concept, sparsity, uncertainty."""

    def __init__(
        self,
        concept_weight: float = 0.2,
        sparsity_weight: float = 2e-4,
        uncertainty_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.concept_weight = concept_weight
        self.sparsity_weight = sparsity_weight
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        concept_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = labels.float()
        cls_loss = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
        sparsity_loss = outputs["effective_concepts"].abs().mean()

        with torch.no_grad():
            confidence = (outputs["prob_ai"].detach() - 0.5).abs() * 2.0
            uncertainty_target = 1.0 - confidence
        uncertainty_loss = F.binary_cross_entropy_with_logits(outputs["uncertainty_logits"], uncertainty_target)

        concept_loss = torch.zeros((), device=labels.device)
        if concept_targets is not None:
            concept_loss = F.binary_cross_entropy_with_logits(outputs["concept_logits"], concept_targets.float())

        total = (
            cls_loss
            + self.concept_weight * concept_loss
            + self.sparsity_weight * sparsity_loss
            + self.uncertainty_weight * uncertainty_loss
        )
        metrics = {
            "loss": total.detach(),
            "loss_cls": cls_loss.detach(),
            "loss_concept": concept_loss.detach(),
            "loss_sparsity": sparsity_loss.detach(),
            "loss_uncertainty": uncertainty_loss.detach(),
        }
        return total, metrics


def sparsity_for_nec(nec: int) -> float:
    if nec <= 5:
        return 1e-4
    if nec <= 10:
        return 2e-4
    return 5e-4
