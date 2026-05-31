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
3. The predicted class and confidence score are shown below the form.

> **Disclaimer:** This tool is for educational purposes only. Always consult a qualified healthcare professional for medical diagnosis.
