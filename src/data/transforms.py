"""Transform ảnh cho train và đánh giá."""

from __future__ import annotations

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224):
    """Augmentation nhẹ cho tập train."""

    try:
        from torchvision import transforms as T
    except ImportError as exc:
        raise ImportError("Cài torchvision để dùng transform: pip install torchvision") from exc

    # Train dùng augmentation ngẫu nhiên để model học bền hơn.
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.01),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
            T.ToTensor(),
            # Normalize theo ImageNet để phù hợp với nhiều backbone pretrained.
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224):
    """Transform cố định cho val/test."""

    try:
        from torchvision import transforms as T
    except ImportError as exc:
        raise ImportError("Cài torchvision để dùng transform: pip install torchvision") from exc

    # Val/test không dùng random augmentation để kết quả đánh giá ổn định.
    return T.Compose(
        [
            T.Resize(image_size + 32),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
