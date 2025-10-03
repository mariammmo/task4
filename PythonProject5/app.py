import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# إعدادات Flask
app = Flask(__name__)

UPLOAD_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# الامتدادات المسموحة
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}

# تحميل الموديل
model = load_model("Flood_Detection_final.h5")

# حجم الصورة المطلوب (غيريه على حسب تدريب الموديل)
IMG_SIZE = (128, 128)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    """قراءة الصورة TIFF أو PNG وتجهيزها للموديل"""
    img = Image.open(filepath)

    # لو TIFF multi-channel ناخد أول layer
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W, C)
    return img


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # تجهيز الصورة للموديل
        img = preprocess_image(filepath)

        # تنبؤ
        prediction = model.predict(img)

        # لو الموديل segmentation (بيطلع mask):
        if len(prediction.shape) == 4:
            mask = (prediction[0] > 0.5).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask.squeeze())
            result_path = os.path.join(app.config["UPLOAD_FOLDER"], "result_" + filename + ".png")
            mask_img.save(result_path)
            return render_template("index.html", uploaded_image=filepath, result_img=result_path)

        # لو الموديل classification (مثلاً Flood / No Flood)
        else:
            label = "Flood" if prediction[0][0] > 0.5 else "No Flood"
            return render_template("index.html", uploaded_image=filepath, label=label)

    return "File type not allowed"


if __name__ == "__main__":
    app.run(debug=True)
