import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

trained_model = None
class_names = [
    'Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot',
    'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 'Mite',
    'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust'
]



class ResNet50V2(nn.Module):
    def __init__(self, num_classes=15):
        super(ResNet50V2, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def get_model():
    global trained_model
    if trained_model is None:
        print("Loading model for the first time...")
        model = ResNet50V2()
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_ResNet50V2.pth")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.eval()
        trained_model = model
        print("Model loaded successfully.")
    return trained_model


def predict(image_path):
    """
    Predict the wheat disease from an image path.
    Logs success or error details to console (Render dashboard).
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        model = get_model():
        with torch.no_grad():
            output = trained_model(image_tensor)
            _, predicted_class = torch.max(output, 1)
            result = class_names[predicted_class.item()]
            print(f"[INFO] Prediction result: {result}")
            return result

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return f"Prediction error: {e}"
