# Skin Cancer Detection Website

A Flask web application that classifies skin lesion images into one of seven diagnostic categories using a fine-tuned EfficientNet-B0 model.

## Supported Classes

| Code   | Description                              |
|--------|------------------------------------------|
| akiec  | Actinic Keratoses / Intraepithelial Carcinoma |
| bcc    | Basal Cell Carcinoma                     |
| bkl    | Benign Keratosis-like Lesions            |
| df     | Dermatofibroma                           |
| mel    | Melanoma                                 |
| nv     | Melanocytic Nevi                         |
| vasc   | Vascular Lesions                         |

## Requirements

- Python 3.8+
- A trained model weights file (`.pth`)

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd Skin-Cancer-Detection-Website
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set the path to your trained model weights via an environment variable:
   ```bash
   export MODEL_PATH=/path/to/your/model.pth   # Windows: set MODEL_PATH=C:\path\to\model.pth
   ```

## Running the App

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

For production deployment with gunicorn:
```bash
gunicorn app:app
```

## Usage

1. Click the upload area or drag and drop an image (JPG or PNG).
2. Click **Analyze Image**.
3. The predicted class, confidence score, risk level, and a full probability breakdown across all classes are shown below the form.

### Batch analysis

Visit `/batch` to upload several images at once. Each image is classified independently and the results are shown in a table, which can be exported to CSV from the browser.

### JSON API

For programmatic access, POST images directly and get JSON back:

- `POST /api/predict` — form field `image` (single file) &rarr; `{label, description, confidence, risk, all_probabilities}`
- `POST /api/batch` — one or more `images` fields &rarr; `{results: [...]}`, one entry per file

Example:
```bash
curl -F "image=@lesion.jpg" http://127.0.0.1:5000/api/predict
```

> **Disclaimer:** This tool is for educational purposes only. Always consult a qualified healthcare professional for medical diagnosis.
