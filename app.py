import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

IMG_WIDTH, IMG_HEIGHT = 224, 224
NUM_CLASSES = 7
class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
class_descriptions = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

# Clinical risk categories to help users understand urgency
RISK_LEVELS = {
    "akiec": "high",
    "bcc":   "high",
    "bkl":   "low",
    "df":    "low",
    "mel":   "high",
    "nv":    "low",
    "vasc":  "medium",
}

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

device = torch.device("cpu")
model_path = os.environ.get("MODEL_PATH", r"/home/eshaan/skin-disease-app/skin_disease_classification_model(1).pth")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Linear(m.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device)
    m.eval()
    return m


try:
    model = load_model()
except Exception as e:
    print(f"Warning: could not load model from '{model_path}': {e}")
    model = None

data_transforms = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_inference(image_file):
    """Run model inference and return label, confidence, risk, and all class probabilities.

    Raises ValueError for unreadable images. Assumes model is not None.
    """
    try:
        image = Image.open(image_file).convert("RGB")
    except Exception:
        raise ValueError("Could not read the uploaded file. Please upload a valid image.")

    tensor = data_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    predicted_label = class_names[pred_idx]
    all_probs = [
        {
            "label": class_names[i],
            "description": class_descriptions[class_names[i]],
            "probability": round(probs[i].item() * 100, 2),
            "risk": RISK_LEVELS[class_names[i]],
        }
        for i in range(NUM_CLASSES)
    ]
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "label": predicted_label,
        "description": class_descriptions[predicted_label],
        "confidence": round(probs[pred_idx].item() * 100, 2),
        "risk": RISK_LEVELS[predicted_label],
        "all_probabilities": all_probs,
    }


@app.errorhandler(413)
def request_entity_too_large(_):
    return render_template(
        "index.html",
        label="", description="", confidence="", risk="", all_probs=None,
        error="File is too large. Please upload an image smaller than 16 MB.",
    ), 413


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        def err(msg):
            return render_template(
                "index.html",
                label="", description="", confidence="", risk="", all_probs=None,
                error=msg,
            )

        if model is None:
            return err("Model is not loaded. Please set the MODEL_PATH environment variable.")

        image_file = request.files.get("image")
        if not image_file or image_file.filename == "":
            return err("No file selected. Please choose an image to upload.")

        if not allowed_file(image_file.filename):
            return err("Unsupported file type. Please upload a JPG or PNG image.")

        try:
            result = run_inference(image_file)
        except ValueError as e:
            return err(str(e))

        return render_template(
            "index.html",
            label=result["label"],
            description=result["description"],
            confidence=f"{result['confidence']:.2f}",
            risk=result["risk"],
            all_probs=result["all_probabilities"],
            error="",
        )

    return render_template(
        "index.html",
        label="", description="", confidence="", risk="", all_probs=None, error="",
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: classify a single skin lesion image.

    Request: multipart/form-data with field 'image' (JPG or PNG).
    Response: JSON with label, description, confidence, risk, and all_probabilities.
    """
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 503

    image_file = request.files.get("image")
    if not image_file or image_file.filename == "":
        return jsonify({"error": "No file provided."}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG or PNG."}), 400

    try:
        result = run_inference(image_file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    """Batch analysis page: upload multiple images, get a results table with CSV export."""
    if request.method == "GET":
        return render_template("batch.html", results=None, error="")

    if model is None:
        return render_template(
            "batch.html", results=None,
            error="Model is not loaded. Please set the MODEL_PATH environment variable.",
        )

    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return render_template("batch.html", results=None,
                               error="No files selected. Please choose at least one image.")

    results = []
    for f in files:
        if not f or f.filename == "":
            continue
        entry = {
            "filename": f.filename,
            "label": "", "description": "", "confidence": "", "risk": "", "error": "",
        }
        if not allowed_file(f.filename):
            entry["error"] = "Unsupported file type"
        else:
            try:
                r = run_inference(f)
                entry["label"] = r["label"]
                entry["description"] = r["description"]
                entry["confidence"] = f"{r['confidence']:.2f}"
                entry["risk"] = r["risk"]
            except ValueError as e:
                entry["error"] = str(e)
        results.append(entry)

    return render_template("batch.html", results=results, error="")


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """JSON API: classify multiple skin lesion images in one request.

    Request: multipart/form-data with one or more 'images' fields.
    Response: JSON with a 'results' list, one entry per image.
    """
    if model is None:
        return jsonify({"error": "Model is not loaded."}), 503

    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files provided."}), 400

    results = []
    for f in files:
        if not f or f.filename == "":
            continue
        if not allowed_file(f.filename):
            results.append({"filename": f.filename, "error": "Unsupported file type"})
            continue
        try:
            r = run_inference(f)
            r["filename"] = f.filename
            results.append(r)
        except ValueError as e:
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True)
