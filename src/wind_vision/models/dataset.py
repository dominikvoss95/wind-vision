"""Wind speed regression dataset backed by CSV labels."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# Precise crop on the center of the lake to maximize wave detail
# and naturally exclude shoreline/overlays. (x1, y1, x2, y2)
_WATER_CROP = (350, 470, 900, 600) 


class WindDataset(Dataset):
    """Pairs webcam images with measured wind speeds for supervised learning.

    Each sample returns the cropped water region and the corresponding
    wind speed in knots as a single-element float tensor.
    """

    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples: list[dict] = []

        with open(csv_file, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                raw = row["wind_kts"].strip()
                if raw.isdigit():
                    self.samples.append({
                        "image": row["image"],
                        "wind_kts": float(raw),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(self.img_dir / item["image"]).convert("RGB")
        image = image.crop(_WATER_CROP)

        # Mask the remaining sliver of the weather overlay (0 to 50px wide in this crop)
        mask = Image.new("RGB", (60, 130), (0, 0, 0))
        image.paste(mask, (0, 0))

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([item["wind_kts"]], dtype=torch.float32)
        return image, label
