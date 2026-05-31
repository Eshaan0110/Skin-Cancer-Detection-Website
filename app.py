import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, render_template, make_response

app = Flask(__name__)

IMG_WIDTH, IMG_HEIGHT = 224, 224
NUM_CLASSES = 7
class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

device = torch.device("cpu")
model_path = os.environ.get("MODEL_PATH", r"/home/eshaan/skin-disease-app/skin_disease_classification_model(1).pth")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    print(f"Warning: could not load model from '{model_path}': {e}")
    model = None

data_transforms = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if model is None:
            return render_template("index.html", prediction="", error="Model is not loaded. Please set the MODEL_PATH environment variable.")

        image_file = request.files.get("image")
        if not image_file or image_file.filename == "":
            return render_template("index.html", prediction="", error="No file selected. Please choose an image to upload.")

        if not allowed_file(image_file.filename):
            return render_template("index.html", prediction="", error="Unsupported file type. Please upload a JPG or PNG image.")

        try:
            image = Image.open(image_file).convert("RGB")
        except (UnidentifiedImageError, Exception):
            return render_template("index.html", prediction="", error="Could not read the uploaded file. Please upload a valid image.")

        image = data_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probabilities, 1)
            predicted_label = class_names[preds[0].item()]
            confidence = confidences[0].item() * 100

        return render_template("index.html", prediction=f"Predicted: {predicted_label}, Confidence: {confidence:.2f}%", error="")

    return render_template("index.html", prediction="", error="")

if __name__ == "__main__":
    app.run(debug=True)
