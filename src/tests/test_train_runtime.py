from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from src.model.batch import unpack_batch
from src.model.evaluate import evaluate_binary_classifier
from src.model.loss import DetectionLoss
from src.model.train import _loader_exists, build_grad_scaler, build_train_loaders, train_one_epoch


class NoLenIterableDataset(IterableDataset):
    def __iter__(self):
        yield torch.randn(3, 8, 8), torch.tensor(1.0)


class TinyDetector(torch.nn.Module):
    def __init__(self, concept_count: int = 15) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(1, 1)
        self.concept_mapper = torch.nn.Linear(1, concept_count)
        self.uncertainty = torch.nn.Linear(1 + concept_count, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        logits = self.classifier(pooled).squeeze(1)
        concept_logits = self.concept_mapper(pooled)
        concepts = torch.sigmoid(concept_logits)
        uncertainty_logits = self.uncertainty(torch.cat([pooled, concepts], dim=1)).squeeze(1)
        return {
            "logits": logits,
            "prob_ai": torch.sigmoid(logits),
            "concept_logits": concept_logits,
            "concepts": concepts,
            "effective_concepts": concepts,
            "uncertainty_logits": uncertainty_logits,
        }


class TrainRuntimeTest(unittest.TestCase):
    def test_unpack_batch_supports_tuple_and_dict(self) -> None:
        device = torch.device("cpu")
        images = torch.randn(2, 3, 8, 8)
        labels = torch.tensor([0, 1])

        tuple_images, tuple_labels = unpack_batch((images, labels), device)
        dict_images, dict_labels = unpack_batch({"image": images, "label": labels}, device)

        self.assertEqual(tuple(tuple_images.shape), (2, 3, 8, 8))
        self.assertEqual(tuple(tuple_labels.shape), (2,))
        self.assertTrue(torch.equal(tuple_images, dict_images))
        self.assertTrue(torch.equal(tuple_labels, dict_labels))

    def test_loader_exists_does_not_call_len_on_iterable_loader(self) -> None:
        loader = DataLoader(NoLenIterableDataset(), batch_size=1)

        self.assertTrue(_loader_exists(loader))
        with self.assertRaises(TypeError):
            len(loader)

    def test_train_one_epoch_runs_without_runtime_error(self) -> None:
        dataset = TensorDataset(torch.randn(4, 3, 16, 16), torch.tensor([0.0, 1.0, 0.0, 1.0]))
        loader = DataLoader(dataset, batch_size=2)
        model = TinyDetector()
        criterion = DetectionLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = build_grad_scaler(enabled=False)

        metrics = train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=torch.device("cpu"),
            amp_dtype=None,
            grad_accum=1,
            epoch=0,
            is_main=False,
        )

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["loss"], 0.0)

    def test_evaluate_runs_without_runtime_error(self) -> None:
        dataset = TensorDataset(torch.randn(4, 3, 16, 16), torch.tensor([0.0, 1.0, 0.0, 1.0]))
        loader = DataLoader(dataset, batch_size=2)
        metrics = evaluate_binary_classifier(TinyDetector(), loader, torch.device("cpu"))

        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("precision_ai", metrics)
        self.assertIn("specificity_real", metrics)
        self.assertIn("predicted_ai_rate", metrics)
        self.assertIn("f1_ai", metrics)

    def test_evaluate_respects_max_steps(self) -> None:
        dataset = TensorDataset(torch.randn(6, 3, 16, 16), torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
        loader = DataLoader(dataset, batch_size=2)
        metrics = evaluate_binary_classifier(TinyDetector(), loader, torch.device("cpu"), max_steps=1)

        self.assertEqual(metrics["false_positive"] + metrics["false_negative"] + metrics["accuracy"] * 2, 2)

    def test_evaluate_respects_max_samples(self) -> None:
        dataset = TensorDataset(torch.randn(6, 3, 16, 16), torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
        loader = DataLoader(dataset, batch_size=4)
        metrics = evaluate_binary_classifier(TinyDetector(), loader, torch.device("cpu"), max_samples=3)

        self.assertEqual(metrics["sample_count"], 3)

    def test_build_train_loaders_uses_new_data_api(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                for label, color in (("real", (32, 96, 160)), ("ai", (180, 60, 80))):
                    for index in range(2):
                        path = root / split / label / f"{label}_{index}.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (48, 48), color=color).save(path)

            loaders, sampler = build_train_loaders(
                data_root=root,
                batch_size=2,
                image_size=32,
                num_workers=0,
                distributed=False,
                pin_memory=False,
            )
            images, labels, metadata = next(iter(loaders["train"]))

        self.assertIsNone(sampler)
        self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertIn("path", metadata)
        self.assertIn("val", loaders)

    def test_build_train_loaders_keeps_small_train_batch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, color in (("real", (32, 96, 160)), ("ai", (180, 60, 80))):
                path = root / "train" / label / f"{label}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 48), color=color).save(path)

            loaders, _ = build_train_loaders(
                data_root=root,
                batch_size=8,
                image_size=32,
                num_workers=0,
                distributed=False,
                pin_memory=False,
            )
            batches = list(loaders["train"])

        self.assertEqual(len(batches), 1)
        self.assertEqual(tuple(batches[0][0].shape), (2, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
