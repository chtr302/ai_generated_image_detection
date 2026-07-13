from __future__ import annotations

import unittest

import torch

from src.model.architectures.detector import build_detector


class ForwardShapeTest(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = build_detector(nec=5, concept_count=15, image_size=128)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 128, 128))

        self.assertEqual(tuple(output["prob_ai"].shape), (1,))
        self.assertEqual(tuple(output["concepts"].shape), (1, 15))
        self.assertEqual(tuple(output["effective_concepts"].shape), (1, 15))
        self.assertEqual(tuple(output["heatmap"].shape), (1, 1, 128, 128))
        self.assertEqual(output["features"]["spatial"].shape[1], 256)
        self.assertEqual(output["features"]["frequency"].shape[1], 256)
        self.assertEqual(output["features"]["wavelet"].shape[1], 256)


if __name__ == "__main__":
    unittest.main()
