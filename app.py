# app.py
import json
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (256, 256)  # khớp train
APP_DIR = Path(__file__).parent

app = Flask(__name__, static_url_path="/static", static_folder="static", template_folder="templates")

# Load model + labels khi khởi động
MODEL = keras.models.load_model(str(APP_DIR / "model.h5"))
with open(APP_DIR / "labels.json", "r", encoding="utf-8") as f:
    LABELS = json.load(f)

def humanize(label: str) -> str:
    # đổi '000200' -> '200.000 ₫' (nếu là mệnh giá). '000000' coi như 'Unknown/Background'
    try:
        v = int(label)
        return f"{v:,}".replace(",", ".") + " ₫"
    except:
        return label

def preprocess_pil(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    x = np.array(img).astype("float32")
    x = preprocess_input(x)          # !!! như lúc train EfficientNet
    x = np.expand_dims(x, 0)         # [1, H, W, 3]
    return x

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    img = Image.open(file.stream)
    x = preprocess_pil(img)
    probs = MODEL.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    return jsonify({
        "raw_label": LABELS[top],
        "label": humanize(LABELS[top]),
        "confidence": float(probs[top]),
        "probs": {LABELS[i]: float(p) for i, p in enumerate(probs)}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
