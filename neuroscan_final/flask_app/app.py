"""
NeuroScan AI — Flask Prediction App (Predict-Only)
Run: python app.py
Visit: http://127.0.0.1:5000
"""
import os, io, base64, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg"}
IMG_SIZE = 64
CLASSES  = ["Mild Dementia", "Moderate Dementia", "Non Demented"]

CLASS_META = {
    "Mild Dementia":     {"icon": "🟡", "color": "#F39C12", "badge": "warning",
                          "desc": "Mild dementia detected. Memory and thinking issues are noticeable. Consult a neurologist."},
    "Moderate Dementia": {"icon": "🔴", "color": "#E74C3C", "badge": "danger",
                          "desc": "Moderate dementia detected. Significant cognitive decline. Immediate medical evaluation is advised."},
    "Non Demented":      {"icon": "🟢", "color": "#2ECC71", "badge": "success",
                          "desc": "No signs of dementia detected. Brain appears normal."},
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_model.h5")
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[NeuroScan] Model loaded: {MODEL_PATH}")
    else:
        print(f"[NeuroScan] WARNING: No model at {MODEL_PATH}")
        print("[NeuroScan] Run: python train_scripts/train_model.py  first.")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0), arr

def confidence_chart(probs):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ["#F39C12", "#E74C3C", "#2ECC71"]
    short  = ["Mild\nDementia", "Moderate\nDementia", "Non\nDemented"]
    bars   = ax.barh(short, probs * 100, color=colors, edgecolor="none", height=0.5)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=10,
                fontweight="bold", color="#333")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=9, color="#555")
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333")
    ax.spines["bottom"].set_color("#333")
    ax.grid(axis="x", alpha=0.15, color="white")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    chart = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return chart

def make_gradcam(img_array, model):
    try:
        # Find last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break
        if not last_conv:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            pred_idx = tf.argmax(preds[0])
            target   = preds[:, pred_idx]
        grads   = tape.gradient(target, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        orig = img_array[0]
        hm   = np.uint8(255 * heatmap)
        hm   = np.array(Image.fromarray(hm).resize((IMG_SIZE, IMG_SIZE)))
        cm   = plt.get_cmap("jet")(hm)[:, :, :3]
        superimposed = np.clip(0.45 * cm + 0.55 * orig, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        for ax in axes: ax.set_facecolor("#0f1117")
        fig.patch.set_facecolor("#0f1117")

        axes[0].imshow(orig);              axes[0].set_title("Original", color="white"); axes[0].axis("off")
        axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Heatmap",  color="white"); axes[1].axis("off")
        axes[2].imshow(superimposed);       axes[2].set_title("Grad-CAM", color="white"); axes[2].axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        result = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return result
    except Exception as e:
        print(f"[Grad-CAM] Failed: {e}")
        return None

# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", model_ready=(model is not None))

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Upload PNG or JPG."}), 400
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_scripts/train_model.py first."}), 503

    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    img_arr, img_rgb = preprocess(save_path)
    probs      = model.predict(img_arr, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])
    meta       = CLASS_META[pred_class]

    chart_b64  = confidence_chart(probs)
    gradcam_b64 = make_gradcam(img_arr, model)

    all_probs  = {CLASSES[i]: f"{probs[i]*100:.1f}" for i in range(len(CLASSES))}

    return render_template("result.html",
        filename    = filename,
        pred_class  = pred_class,
        confidence  = f"{confidence*100:.1f}",
        icon        = meta["icon"],
        color       = meta["color"],
        badge       = meta["badge"],
        desc        = meta["desc"],
        chart_b64   = chart_b64,
        gradcam_b64 = gradcam_b64,
        all_probs   = all_probs,
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    img_arr, _ = preprocess(save_path)
    probs      = model.predict(img_arr, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    return jsonify({
        "prediction"     : CLASSES[pred_idx],
        "confidence"     : float(probs[pred_idx]),
        "probabilities"  : {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    })

if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=10000)
