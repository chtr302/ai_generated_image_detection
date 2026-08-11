from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import types
import unittest

from src.data import (
    DataLoaderConfig,
    build_dataloaders,
    build_datasets,
    discover_image_records,
    normalize_label,
    split_records,
)
from src.data.dataset import HuggingFaceImageDataset, StreamingHuggingFaceImageDataset


# Kiem tra dependency tuy chon cho test DataLoader that.
def _optional_training_dependencies_available() -> bool:
    try:
        import PIL  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return True


def _touch_fake_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _save_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 40), color=color).save(path)


class DatasetUtilsTest(unittest.TestCase):
    # Test quet folder da co train/val va gan label dung.
    def test_discovers_labels_and_existing_splits(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val"):
                for label in ("real", "ai"):
                    _touch_fake_image(root / split / label / f"{label}.jpg")

            records = discover_image_records(root)

        self.assertEqual(len(records), 4)
        self.assertEqual({record.split for record in records}, {"train", "val"})
        self.assertEqual({record.label for record in records}, {0, 1})

    # Test tu chia train/val/test khi data chua chia san.
    def test_splits_unsplit_records_stratified(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ("real", "ai"):
                for index in range(5):
                    _touch_fake_image(root / label / f"{label}_{index}.png")

            records = discover_image_records(root)
            split_records_ = split_records(records, seed=7)

        self.assertEqual(len(split_records_), 10)
        self.assertLessEqual(
            {record.split for record in split_records_},
            {"train", "val", "test"},
        )

    # Test cac ten label khac nhau duoc dua ve 0/1.
    def test_normalizes_label_aliases(self) -> None:
        self.assertEqual(normalize_label("authentic"), 0)
        self.assertEqual(normalize_label("generated"), 1)

    @unittest.skipUnless(
        _optional_training_dependencies_available(),
        "Pillow, torch, and torchvision are required for DataLoader tests",
    )
    def test_builds_dataloader_batch_from_real_images(self) -> None:
        # Test doc anh that, transform thanh tensor va gom batch.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label, color in (("real", (32, 96, 160)), ("ai", (180, 60, 80))):
                for index in range(2):
                    _save_rgb_image(root / "val" / label / f"{label}_{index}.png", color)

            config = DataLoaderConfig(data_root=root, image_size=32, batch_size=2)
            loaders = build_dataloaders(config)
            images, labels, metadata = next(iter(loaders["val"]))

        self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertLessEqual(set(labels.tolist()), {0, 1})
        self.assertEqual(len(metadata["path"]), 2)
        self.assertEqual(metadata["split"], ["val", "val"])


    def test_huggingface_image_dataset_reads_defactify_fields(self) -> None:
        from PIL import Image
        from torchvision import transforms as T

        rows = [
            {
                "Image": Image.new("RGB", (24, 24), color=(120, 80, 40)),
                "Label_A": 1,
                "Label_B": 2,
                "Caption": "synthetic sample",
            }
        ]
        dataset = HuggingFaceImageDataset(
            rows,
            transform=T.Compose([T.Resize((16, 16)), T.ToTensor()]),
            return_metadata=True,
        )
        image, label, metadata = dataset[0]

        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertEqual(label, 1)
        self.assertEqual(metadata["source"], "2")
        self.assertEqual(metadata["caption"], "synthetic sample")

    @unittest.skipUnless(
        _optional_training_dependencies_available(),
        "Pillow, torch, and torchvision are required for Hugging Face loader tests",
    )
    def test_build_datasets_uses_defactify_huggingface_streaming_splits(self) -> None:
        from PIL import Image

        calls = []

        def fake_load_dataset(dataset_id, *args, **kwargs):
            calls.append((dataset_id, args, kwargs))
            return [
                {
                    "Image": Image.new("RGB", (24, 24), color=(10, 20, 30)),
                    "Label_A": 0,
                    "Label_B": 0,
                    "Caption": f"{kwargs['split']} sample",
                }
            ]

        original_module = sys.modules.get("datasets")
        sys.modules["datasets"] = types.SimpleNamespace(load_dataset=fake_load_dataset)
        try:
            datasets = build_datasets(
                DataLoaderConfig(
                    hf_dataset_id="Rajarshi-Roy-research/Defactify_Image_Dataset",
                    hf_cache_dir="/tmp/hf_cache",
                    image_size=16,
                )
            )
        finally:
            if original_module is None:
                sys.modules.pop("datasets", None)
            else:
                sys.modules["datasets"] = original_module

        self.assertEqual(set(datasets), {"train", "val", "test"})
        self.assertEqual([call[2]["split"] for call in calls], ["train", "validation", "test"])
        self.assertTrue(all(call[2]["streaming"] for call in calls))
        self.assertIsInstance(datasets["train"], StreamingHuggingFaceImageDataset)
        image, label, metadata = next(iter(datasets["val"]))
        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertEqual(label, 0)
        self.assertEqual(metadata["caption"], "validation sample")



if __name__ == "__main__":
    unittest.main()



