import os
import json
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # REQUIRED for cloud

from gradcam import grad_cam_densenet, detect_orientation
from utils import preprocess_image

# ---------------- CONFIG ----------------
IMG_SIZE = (224, 224)
UPLOAD_FOLDER = "static/outputs"
MODEL_PATH = "dfu_densenet_ce_model.keras"   # rename file (NO spaces)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- LAZY MODEL LOADING ----------------
model = None
class_map = None

def get_model():
    global model, class_map
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open("class_map.json") as f:
            class_map = json.load(f)
    return model, class_map


# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    try:
        if request.method == "POST":
            file = request.files.get("image")
            if not file:
                return render_template("index.html", error="No image uploaded")

            image = Image.open(file).convert("RGB")
            img_arr = preprocess_image(image, IMG_SIZE)

            preds = model.predict(img_arr, verbose=0)[0]
            grade_idx = int(np.argmax(preds))
            grade = grade_idx + 1
            confidence = float(preds[grade_idx] * 100)

            heatmap = grad_cam_densenet(model, img_arr)
            orientation = detect_orientation(heatmap)

            heatmap = cv2.resize(heatmap, IMG_SIZE)
            heatmap_col = cv2.applyColorMap(
                np.uint8(255 * heatmap), cv2.COLORMAP_JET
            )

            overlay = heatmap_col * 0.4 + np.array(image.resize(IMG_SIZE))

            output_path = os.path.join(UPLOAD_FOLDER, "gradcam_result.png")
            cv2.imwrite(
                output_path,
                cv2.cvtColor(overlay.astype("uint8"), cv2.COLOR_RGB2BGR)
            )

            return render_template(
                "index.html",
                grade=grade,
                confidence=f"{confidence:.2f}",
                orientation=orientation,
                image_path=output_path
            )

        return render_template("index.html")

    except Exception as e:
        print("🔥 ERROR:", e)
        return render_template(
            "index.html",
            error=str(e)
        )


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

