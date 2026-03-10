"""Evaluate a trained wind model against the labelled dataset."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms

from wind_vision.models.dataset import WindDataset

logger = logging.getLogger("wind_vision.evaluate")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    """Recreate the architecture and load saved weights."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate(
    csv_file: str = "data/processed_wind.csv",
    img_dir: str = "data/raw/webcam",
    weights: str = "models/wind_model.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = WindDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    # use the same 80/20 split as training (fixed seed for reproducibility)
    generator = torch.Generator().manual_seed(42)
    train_n = int(0.8 * len(dataset))
    val_n = len(dataset) - train_n
    _, val_ds = random_split(dataset, [train_n, val_n], generator=generator)

    loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    model = load_model(weights, device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    preds = torch.cat(all_preds).squeeze()
    labels = torch.cat(all_labels).squeeze()

    # ── metrics ──────────────────────────────────────────────────
    errors = preds - labels
    abs_errors = errors.abs()

    mae = abs_errors.mean().item()
    rmse = (errors ** 2).mean().sqrt().item()
    max_err = abs_errors.max().item()

    # percentage of predictions within ±2 kts of the true value
    within_2 = (abs_errors <= 2.0).float().mean().item() * 100

    print("\n" + "=" * 50)
    print(f"  Validation samples:   {len(labels)}")
    print(f"  MAE  (avg error):     {mae:.2f} kts")
    print(f"  RMSE:                 {rmse:.2f} kts")
    print(f"  Max error:            {max_err:.1f} kts")
    print(f"  Within ±2 kts:        {within_2:.1f}%")
    print("=" * 50)

    # show some example predictions
    print("\n  Sample predictions (true → predicted):\n")
    indices = torch.randperm(len(labels))[:15]
    for i in indices:
        true_val = labels[i].item()
        pred_val = preds[i].item()
        diff = pred_val - true_val
        marker = "✓" if abs(diff) <= 2.0 else "✗"
        print(f"    {true_val:5.1f} kts → {pred_val:5.1f} kts  (Δ {diff:+.1f})  {marker}")

    print()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    evaluate()
