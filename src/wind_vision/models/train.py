"""Training loop for the wind speed regression model."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms

from wind_vision.models.dataset import WindDataset

logger = logging.getLogger("wind_vision.train")

# ImageNet statistics used by all torchvision pretrained weights.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(device: torch.device) -> nn.Module:
    """Load ResNet-18 with pretrained weights, swap FC head for regression."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)


def train_model(
    csv_file: str,
    img_dir: str,
    num_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = WindDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    if len(dataset) == 0:
        logger.error("Empty dataset — check CSV and image directory.")
        return

    train_n = int(0.8 * len(dataset))
    val_n = len(dataset) - train_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_n else None

    logger.info("Samples: %d (train=%d, val=%d)", len(dataset), train_n, val_n)

    model = build_model(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= train_n

        # --- validate ---
        val_msg = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    val_loss += criterion(model(images), labels).item() * images.size(0)
            val_loss /= val_n
            val_msg = f" | val={val_loss:.4f}"

        logger.info("Epoch %d/%d  train=%.4f%s", epoch, num_epochs, train_loss, val_msg)

    # persist weights
    out = Path("models")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "wind_model.pth")
    logger.info("Saved model → %s", out / "wind_model.pth")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train_model(
        csv_file="data/processed_wind.csv",
        img_dir="data/raw/webcam",
    )
