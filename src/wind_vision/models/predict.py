"""Run inference with the trained wind model on a single image."""

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import torch.nn as nn
from torchvision import models

# Same ImageNet statistics and water crop as used in training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Focus on center lake area (x1, y1, x2, y2)
_WATER_CROP = (350, 470, 900, 600)


def predict_wind(image_path: str, model_path: str = "models/wind_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model architecture
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess the image
    img = Image.open(image_path).convert("RGB")
    img = img.crop(_WATER_CROP)

    # Mask overlay sliver
    mask = Image.new("RGB", (60, 130), (0, 0, 0))
    img.paste(mask, (0, 0))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return max(0.0, prediction)  # Wind speed cannot be negative


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m wind_vision.models.predict <image_path>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    res = predict_wind(img_path)
    print(f"\nPredicted Wind Speed: {res:.1f} kts\n")
