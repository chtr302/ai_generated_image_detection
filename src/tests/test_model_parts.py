from __future__ import annotations

import unittest

import torch

from src.forensics.multidomain import MultiDomainPreprocessor
from src.model.architectures.branch_encoder import BranchEncoder
from src.model.architectures.detector import build_detector
from src.model.architectures.fusion import CrossDomainAttentionFusion


class ModelPartShapeTest(unittest.TestCase):
    def test_multidomain_preprocessor_outputs_three_domains(self) -> None:
        preprocessor = MultiDomainPreprocessor()
        image = torch.randn(2, 3, 128, 128)
        domains = preprocessor(image)

        self.assertEqual(set(domains), {"spatial", "frequency", "wavelet"})
        self.assertEqual(tuple(domains["spatial"].shape), (2, 6, 128, 128))
        self.assertEqual(tuple(domains["frequency"].shape), (2, 3, 128, 128))
        self.assertEqual(tuple(domains["wavelet"].shape), (2, 9, 128, 128))

    def test_branch_encoder_and_fusion_shapes(self) -> None:
        branches = {
            "spatial": BranchEncoder(6)(torch.randn(1, 6, 128, 128)),
            "frequency": BranchEncoder(3)(torch.randn(1, 3, 128, 128)),
            "wavelet": BranchEncoder(9)(torch.randn(1, 9, 128, 128)),
        }
        fused = CrossDomainAttentionFusion()(branches)

        self.assertEqual(tuple(branches["spatial"].shape), (1, 256, 4, 4))
        self.assertEqual(tuple(fused.shape), (1, 256, 4, 4))

    def test_detector_forward_output_contract(self) -> None:
        model = build_detector(nec=5, concept_count=15, image_size=128)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 128, 128))

        self.assertEqual(tuple(output["prob_ai"].shape), (1,))
        self.assertEqual(tuple(output["concepts"].shape), (1, 15))
        self.assertEqual(tuple(output["effective_concepts"].shape), (1, 15))
        self.assertEqual(tuple(output["heatmap"].shape), (1, 1, 128, 128))
        self.assertIn("spatial", output["features"])
        self.assertIn("frequency", output["features"])
        self.assertIn("wavelet", output["features"])


if __name__ == "__main__":
    unittest.main()
