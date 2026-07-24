from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.transforms import build_eval_transform
from src.model.architectures.detector import build_detector
from src.xai.concept_bottleneck import DEFAULT_CONCEPTS
from src.xai.explanation import render_explanation


DEFAULT_IMAGE_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference cho SFW-SwinCBM.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--nec", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    nec: int = 10,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector(nec=nec, concept_count=len(DEFAULT_CONCEPTS), image_size=image_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    transform = build_eval_transform(image_size)
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    probability = float(output["prob_ai"].item())
    prediction = "ai_generated" if probability >= threshold else "real"
    concepts = output["concepts"].squeeze(0).detach().cpu()
    effective = output["effective_concepts"].squeeze(0).detach().cpu()
    top_indices = torch.topk(effective, k=min(5, effective.numel())).indices.tolist()
    top_concepts = [
        {"name": DEFAULT_CONCEPTS[index], "score": round(float(concepts[index]), 4)}
        for index in top_indices
        if float(effective[index]) > 0.0
    ]

    top_names = [item["name"] for item in top_concepts]
    real_probability = 1.0 - probability
    confidence = max(probability, real_probability)
    return {
        "prediction": prediction,
        "ai_probability": round(probability, 4),
        "real_probability": round(real_probability, 4),
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "concepts_top5": top_concepts,
        "effective_concepts": top_names,
        "heatmap": output["heatmap"].squeeze(0).detach().cpu(),
        "explanation": render_explanation(prediction, top_names, probability),
        "fft_explanation": render_fft_explanation(prediction, probability),
        "debug": {"nec": nec, "image_size": image_size},
    }


def render_fft_explanation(prediction: str, probability: float) -> str:
    if prediction == "ai_generated":
        return (
            "FFT doc anh theo mien tan so: thay vi nhin vat the, no nhin cac mau lap lai, "
            "bien nhan tao va nang luong tan so cao/thap bat thuong. Khi cac dau vet nay du manh, "
            f"model tang xac suat AI len {probability:.2%}."
        )
    return (
        "FFT khong thay dau vet tan so bat thuong du manh. Noi cach khac, texture va bien anh "
        f"gan voi cach camera/nen anh that tao ra hon, nen xac suat AI chi la {probability:.2%}."
    )


def main() -> None:
    args = parse_args()
    result = predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        nec=args.nec,
        threshold=args.threshold,
    )
    printable = {key: value for key, value in result.items() if key not in {"heatmap", "debug"}}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
