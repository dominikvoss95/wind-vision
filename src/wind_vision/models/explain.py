"""Visualize model focus using Grad-CAM to see which pixels drive the prediction."""

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from pathlib import Path

# Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_WATER_CROP = (350, 470, 900, 600)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass for the prediction
        self.model.zero_grad()
        output.backward()

        # Weight the channels by the gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU and normalization
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        return cam

def run_explanation(image_path: str, model_path: str = "models/wind_model.pth"):
    device = torch.device("cpu") # Use CPU for stable visualization

    # Load model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess
    orig_img = Image.open(image_path).convert("RGB")
    crop_img = orig_img.crop(_WATER_CROP)
    
    # Mask overlay sliver
    mask = Image.new("RGB", (60, 130), (0, 0, 0))
    crop_img.paste(mask, (0, 0))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    input_tensor = transform(crop_img).unsqueeze(0).requires_grad_(True)

    # Setup Grad-CAM on the last convolutional layer of ResNet18
    target_layer = model.layer4[-1]
    cam_engine = GradCAM(model, target_layer)
    
    heatmap = cam_engine.generate_heatmap(input_tensor)

    # Create visualization
    # Convert PIL crop to CV2 BGR
    vis_img = np.array(crop_img.resize((224, 224)))
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    # Overlay heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    result = cv2.addWeighted(vis_img, 0.6, heatmap_colored, 0.4, 0)

    out_name = "debug_focus_heatmap.png"
    cv2.imwrite(out_name, result)
    print(f"Heatmap saved as: {out_name}")

if __name__ == "__main__":
    import sys
    img_p = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    run_explanation(img_p)
