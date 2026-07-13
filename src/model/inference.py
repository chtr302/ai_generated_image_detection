from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from src.data.transforms import build_eval_transform
from src.model.architectures.detector import build_detector
from src.xai.concept_bottleneck import DEFAULT_CONCEPTS
from src.xai.explanation import render_explanation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference cho SFW-SwinCBM.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--nec", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    image_size: int = 384,
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

    return {
        "prediction": prediction,
        "ai_probability": round(probability, 4),
        "concept_vector": [round(float(value), 4) for value in concepts],
        "concepts_top5": top_concepts,
        "effective_concepts": [item["name"] for item in top_concepts],
        "heatmap": output["heatmap"].squeeze(0).detach().cpu(),
        "explanation": render_explanation(prediction, [item["name"] for item in top_concepts], probability),
        "debug": {"threshold": threshold, "nec": nec},
    }


def main() -> None:
    args = parse_args()
    result = predict_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        nec=args.nec,
        threshold=args.threshold,
    )
    printable = {key: value for key, value in result.items() if key != "heatmap"}
    print(printable)


if __name__ == "__main__":
    main()
