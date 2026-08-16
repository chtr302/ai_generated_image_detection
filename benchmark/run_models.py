from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


LABEL_TO_INT = {"real": 0, "fake": 1}
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(3, 1, 1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"
AI_CLASS_INDEX_ENV = "AIGID_AI_CLASS_INDEX"

XRAYON_MODEL_SPECS = (
    {
        "key": "hybrid",
        "name": "Hybrid XRayon Physical",
        "filename": "hybrid_xrayon_physical.onnx",
        "image_size": 224,
    },
    {
        "key": "rgb",
        "name": "XRayon RGB Only",
        "filename": "xrayon_rgb_only.onnx",
        "image_size": 256,
    },
)

RAW_OUTPUTS = {
    "Hybrid XRayon Physical": "raw_hybrid_xrayon_physical.csv",
    "XRayon RGB Only": "raw_xrayon_rgb_only.csv",
    "UniversalFakeDetect": "raw_universalfakedetect.csv",
}


class UniversalManifestDataset(Dataset):
    def __init__(self, benchmark_root: Path, rows: list[dict[str, str]]) -> None:
        self.benchmark_root = benchmark_root
        self.rows = rows
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(self.benchmark_root / row["relative_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark models and write raw fake_score files.")
    parser.add_argument("--benchmark-root", default="benchmark/data/openfake_1k")
    parser.add_argument("--output-root", default="benchmark/results/openfake_1k")
    parser.add_argument("--model-dir", default=os.getenv("AIGID_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    parser.add_argument("--ai-class-index", type=int, default=int(os.getenv(AI_CLASS_INDEX_ENV, "1")))
    parser.add_argument("--universal-repo-dir", default="benchmark/external/UniversalFakeDetect")
    parser.add_argument("--limit-per-label", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["xrayon", "universal", "all"],
        default=["all"],
        help="'xrayon' runs both ONNX models; 'universal' runs UniversalFakeDetect.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def selected_groups(model_args: list[str]) -> set[str]:
    return {"xrayon", "universal"} if "all" in model_args else set(model_args)


def load_manifest(benchmark_root: Path) -> list[dict[str, str]]:
    with (benchmark_root / "manifest.csv").open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["label"] = row["label"].lower().strip()
        if row["label"] not in LABEL_TO_INT:
            raise ValueError(f"Invalid label: {row['label']}")
    return rows


def select_balanced_subset(rows: list[dict[str, str]], limit_per_label: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for label in ("real", "fake"):
        label_rows = [row for row in rows if row["label"] == label]
        if len(label_rows) < limit_per_label:
            raise ValueError(f"Need {limit_per_label} {label} images, found {len(label_rows)}")
        if limit_per_label == len(label_rows):
            selected.extend(label_rows)
            continue

        indices = np.linspace(0, len(label_rows) - 1, limit_per_label, dtype=int)
        selected.extend(label_rows[int(index)] for index in indices)

    return sorted(selected, key=lambda row: row["sample_id"])


def make_raw_row(row: dict[str, str], fake_score: float) -> dict[str, Any]:
    default_pred = "fake" if fake_score >= 0.5 else "real"
    return {
        "sample_id": row["sample_id"],
        "relative_path": row["relative_path"],
        "true_label": row["label"],
        "default_pred_label": default_pred,
        "fake_score": fake_score,
        "source_model": row.get("model", ""),
        "source_type": row.get("type", ""),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def raw_matches_subset(path: Path, rows: list[dict[str, str]]) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as file:
        raw_rows = list(csv.DictReader(file))
    return [row["sample_id"] for row in raw_rows] == [row["sample_id"] for row in rows]


def preprocess_xrayon_image(image: Image.Image, image_size: int) -> np.ndarray:
    resize_size = image_size + 32
    width, height = image.size
    scale = resize_size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)

    left = max(0, (resized.width - image_size) // 2)
    top = max(0, (resized.height - image_size) // 2)
    cropped = resized.crop((left, top, left + image_size, top + image_size))

    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(array.astype(np.float32), axis=0)


def load_xrayon_sessions(model_dir: Path) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime is not installed. Install it with: pip install onnxruntime") from exc

    sessions: dict[str, Any] = {}
    for spec in XRAYON_MODEL_SPECS:
        path = model_dir / str(spec["filename"])
        if not path.exists():
            raise FileNotFoundError(f"Missing ONNX model: {path}")
        sessions[str(spec["key"])] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sessions


def run_xrayon_batch(
    session: Any,
    spec: dict[str, Any],
    images: list[Image.Image],
    ai_class_index: int,
) -> list[float]:
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    tensor = np.concatenate([preprocess_xrayon_image(image, int(spec["image_size"])) for image in images], axis=0)
    raw_output = session.run([model_output.name], {model_input.name: tensor})[0]
    output_values = np.asarray(raw_output, dtype=np.float32)

    if output_values.ndim != 2:
        raise RuntimeError(f"Output of {spec['name']} is not a batch matrix.")
    if output_values.shape[1] < 2:
        raise RuntimeError(f"Output of {spec['name']} must have at least two classes.")
    if ai_class_index < 0 or ai_class_index >= output_values.shape[1]:
        raise RuntimeError(f"{AI_CLASS_INDEX_ENV}={ai_class_index} is invalid for {output_values.shape[1]} classes.")

    return [float(score) for score in output_values[:, ai_class_index].tolist()]


def run_xrayon(
    benchmark_root: Path,
    rows: list[dict[str, str]],
    output_root: Path,
    model_dir: Path,
    ai_class_index: int,
    batch_size: int,
    force: bool,
) -> None:
    predictions_root = output_root / "predictions"
    output_paths = {
        name: predictions_root / filename
        for name, filename in RAW_OUTPUTS.items()
        if name in {"Hybrid XRayon Physical", "XRayon RGB Only"}
    }
    if not force and all(raw_matches_subset(path, rows) for path in output_paths.values()):
        for path in output_paths.values():
            print(f"Reuse raw score: {path}")
        return

    sessions = load_xrayon_sessions(model_dir)
    raw_rows_by_model: dict[str, list[dict[str, Any]]] = {name: [] for name in output_paths}

    for start in tqdm(range(0, len(rows), batch_size), desc="XRayon ONNX inference"):
        batch_rows = rows[start : start + batch_size]
        images: list[Image.Image] = []
        for row in batch_rows:
            with Image.open(benchmark_root / row["relative_path"]) as image_file:
                images.append(image_file.convert("RGB"))
        for spec in XRAYON_MODEL_SPECS:
            model_name = str(spec["name"])
            fake_scores = run_xrayon_batch(sessions[str(spec["key"])], spec, images, ai_class_index)
            for row, fake_score in zip(batch_rows, fake_scores):
                raw_rows_by_model[model_name].append(make_raw_row(row, fake_score))

    for model_name, output_path in output_paths.items():
        write_rows(output_path, raw_rows_by_model[model_name])


def load_universal_model(repo_dir: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(repo_dir.resolve()))
    from models import get_model  # type: ignore

    model = get_model("CLIP:ViT-L/14")
    ckpt_path = repo_dir / "pretrained_weights" / "fc_weights.pth"
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.fc.load_state_dict(state_dict)
    model.eval()
    return model.to(device)


def run_universalfakedetect(
    benchmark_root: Path,
    rows: list[dict[str, str]],
    output_path: Path,
    batch_size: int,
    device: torch.device,
    repo_dir: Path,
) -> None:
    model = load_universal_model(repo_dir, device)
    dataset = UniversalManifestDataset(benchmark_root, rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    raw_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="UniversalFakeDetect inference"):
            images = images.to(device)
            scores = model(images).sigmoid().flatten().detach().cpu().tolist()
            for index, score in zip(indices.tolist(), scores):
                raw_rows.append(make_raw_row(rows[index], float(score)))

    write_rows(output_path, raw_rows)


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    output_root = Path(args.output_root)
    model_dir = Path(args.model_dir)
    repo_dir = Path(args.universal_repo_dir)
    device = pick_device(args.device)
    groups = selected_groups(args.models)
    rows = select_balanced_subset(load_manifest(benchmark_root), args.limit_per_label)

    predictions_root = output_root / "predictions"
    universal_path = predictions_root / RAW_OUTPUTS["UniversalFakeDetect"]

    if "xrayon" in groups:
        run_xrayon(benchmark_root, rows, output_root, model_dir, args.ai_class_index, args.batch_size, args.force)

    if "universal" in groups:
        if args.force or not raw_matches_subset(universal_path, rows):
            run_universalfakedetect(benchmark_root, rows, universal_path, args.batch_size, device, repo_dir)
        else:
            print(f"Reuse raw score: {universal_path}")


if __name__ == "__main__":
    main()
