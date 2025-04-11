import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from flask import Flask, request, render_template

app = Flask(__name__)  # Fixing _name_ to __name__

# Constants
IMG_WIDTH, IMG_HEIGHT = 224, 224
NUM_CLASSES = 7  # Ensure this matches your dataset
class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Load model
device = torch.device("cpu")  # Force CPU instead of CUDA

model_path = r"D:\Projects\Underdeveloped\skin-disease-classification\models\skin_disease_classification_model_finetuned.pth"

model = efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)

# Load model weights
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

model = model.to(device)
model.eval()

# Data preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image = Image.open(image_file).convert("RGB")
            image = data_transforms(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, preds = torch.max(probabilities, 1)
                predicted_label = class_names[preds[0].item()]
                confidence = confidences[0].item() * 100
                return render_template("index.html", prediction=f'Predicted: {predicted_label}, Confidence: {confidence:.2f}%')

    return render_template("index.html", prediction="")

if __name__ == "__main__":  # Fixing _name_ to __name__
    app.run(debug=True)
