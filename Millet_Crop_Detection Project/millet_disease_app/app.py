import os
import re
import uuid
import base64
import traceback
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
import pillow_avif  # keep import so plugin registers with Pillow if installed

# ------------------ App config ------------------
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXT = {"jpg", "jpeg", "png", "avif"}

# ------------------ Model paths (update if needed) ------------------
disease_model_path = os.path.abspath(
    r"C:\Users\91843\Desktop\Projects\Millet_Crop_Detection Project\millet_disease_app\models\MobileNetV2_millet_model.keras"
)
treatability_model_path = os.path.abspath(
    r"C:\Users\91843\Desktop\Projects\Millet_Crop_Detection Project\millet_disease_app\models\stage_classification_model.keras"
)

# ------------------ Load models ------------------
if not os.path.exists(disease_model_path):
    raise FileNotFoundError(f"❌ ERROR: Disease model not found at {disease_model_path}")
if not os.path.exists(treatability_model_path):
    raise FileNotFoundError(f"❌ ERROR: Treatability model not found at {treatability_model_path}")

# load without compiling (safer cross-version)
disease_model = load_model(disease_model_path, compile=False)
treatability_model = load_model(treatability_model_path, compile=False)

# ------------------ Class labels & solutions ------------------
disease_classes = [
    "Finger (Ragi) Downy", "Finger (Ragi) Mottle", "Finger (Ragi) Smut", "Finger (Ragi) Seedling", "Finger (Ragi) Wilt",
    "Healthy", "Pearl Healthy", "Pearl Rust Disease", "Pearl Downy Mildew",
    "Sorghum (Jowar) Blast", "Sorghum (Jowar) Ergot", "Sorghum (Jowar) Smut",
    "Sorghum (Jowar) Rust", "Sorghum (Jowar) Healthy"
]

disease_solutions = {
    "Finger (Ragi) Downy": "Spray Metalaxyl 8% + Mancozeb 64% WP at 2 g/L water...",
    "Finger (Ragi) Mottle": "Avoid excessive nitrogen fertilizers...",
    "Finger (Ragi) Smut": "Treat seeds with Captan or Thiram @ 2g/kg of seed...",
    "Finger (Ragi) Wilt": "Improve soil drainage to reduce fungal buildup...",
    "Finger (Ragi) Seedling": (
        "Ensure seed treatment with fungicides such as Thiram or Captan (2g/kg seed) "
        "before sowing to protect seedlings. Use well-drained soil and avoid waterlogging. "
        "Maintain proper spacing and apply bio-control agents like Trichoderma to reduce early damping-off."
    ),
    "Healthy": "No action required, your plant is in good condition.",
    "Pearl Healthy": "No action required, your plant is in good condition.",
    "Pearl Rust Disease": "Spray Propiconazole 0.1% or Mancozeb 75 WP...",
    "Pearl Downy Mildew": "Apply Metalaxyl 35% WS as a seed treatment...",
    "Sorghum (Jowar) Blast": "Spray Tricyclazole 0.1% or Carbendazim 0.2%...",
    "Sorghum (Jowar) Ergot": "Apply fungicidal spray of Carbendazim 0.1%...",
    "Sorghum (Jowar) Smut": "Treat seeds with Captan or Thiram @ 3g/kg of seed...",
    "Sorghum (Jowar) Rust": "Use Azoxystrobin 0.1% or Tebuconazole 0.2% sprays...",
    "Sorghum (Jowar) Healthy": "No action required, your plant is in good condition."
}

# ------------------ Chatbot flows ------------------
chatbot_questions = {
    "start": [
        "How to identify millet diseases?",
        "What fertilizers should I use for millets?",
        "How to prevent pest attacks?",
        "What are the best irrigation methods for millets?",
        "Are there government schemes for millet farmers?",
        "Where can I sell my millet produce?",
    ],
    "How to identify millet diseases?": [
        "Upload an image for disease detection.",
        "Common signs of millet diseases?",
        "Are there home remedies for millet diseases?",
    ],
    "What fertilizers should I use for millets?": [
        "Best organic fertilizers for millets?",
        "When to apply fertilizers for maximum yield?",
    ],
    "How to prevent pest attacks?": [
        "Natural pesticides for millet farming?",
        "What are the common millet pests?",
    ],
    "What are the best irrigation methods for millets?": [
        "Drip irrigation vs. flood irrigation?",
        "How often should I water my millet crops?",
    ],
    "Are there government schemes for millet farmers?": [
        "Which subsidies are available for millet farmers?",
        "How to apply for government grants?",
    ],
    "Where can I sell my millet produce?": [
        "Best markets for millet farmers?",
        "How to get better prices for millet?",
    ],
}

chatbot_answers = {
    "How to identify millet diseases?": (
        "You can identify millet diseases through leaf spots, rust, and wilting. "
        "For precise detection, you can use our image-based millet disease detection system. "
        '<a href="https://www.millets.res.in/books/DISEASES_OF_MILLETS.pdf" target="_blank">Read More</a>'
    ),
    "What fertilizers should I use for millets?": (
        "For millet farming, organic manure, NPK fertilizers, and biofertilizers like Azospirillum are effective. "
        "Applying fertilizers at the right growth stage increases yield. "
        '<a href="https://extension.umn.edu/crop-specific-needs/millet-fertilizer-guidelines" target="_blank">Read More</a>'
    ),
    "How to prevent pest attacks?": (
        "To prevent pest attacks, use neem oil, pheromone traps, and crop rotation techniques. "
        "Integrated Pest Management (IPM) is highly recommended for millet farming. "
        '<a href="https://krishi.icar.gov.in/jspui/bitstream/123456789/11165/1/Indian%20Farming%2C%20July%202015.PDF" target="_blank">Read More</a>'
    ),
    "What are the best irrigation methods for millets?": (
        "Millets require less water compared to other grains. Drip irrigation conserves water and enhances yield. "
        "Over-watering can lead to fungal diseases. "
        '<a href="http://milletmiracles.com/general/irrigation-strategies-for-millets-from-traditional-methods-to-modern-innovation/" target="_blank">Read More</a>'
    ),
    "Are there government schemes for millet farmers?": (
        "Yes, the government offers various schemes like MSP, subsidies, and organic farming incentives. "
        "Check the latest schemes for millet farmers. "
        '<a href="https://pib.gov.in/PressReleaseIframePage.aspx?PRID=2082229#:~:text=To%20promote%20the%20use%20of,outlay%20of%20%E2%82%B9800%20crore." target="_blank">Read More</a>'
    ),
    "Where can I sell my millet produce?": (
        "You can sell your produce at local mandis, agricultural co-operatives, or online platforms like eNAM. "
        "Understanding market trends helps in better pricing. "
        '<a href="https://pressroom.icrisat.org/farm-to-fork-an-overview-of-millet-supply-chains-in-india" target="_blank">Read More</a>'
    ),
}

# ------------------ Helpers ------------------
def allowed_file_extension(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXT

def convert_avif_to_jpg(avif_path):
    """Converts an AVIF image to JPEG format using Pillow (PIL)."""
    try:
        img = Image.open(avif_path)
        jpg_path = avif_path.rsplit(".", 1)[0] + ".jpg"
        img.convert("RGB").save(jpg_path, "JPEG")
        return jpg_path
    except Exception as e:
        app.logger.exception("AVIF->JPG conversion failed")
        return None

def open_image_force_rgb(path_or_file):
    """
    Open image, convert to RGB, and return a PIL.Image object.
    Accepts a path string or a file-like object.
    """
    img = Image.open(path_or_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def preprocess_image_pil(pil_img, size=(224, 224)):
    """Return numpy array shaped (1, H, W, 3) normalized to [0,1]."""
    pil_img = pil_img.resize(size)
    arr = np.asarray(pil_img).astype("float32") / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_disease(img_path):
    """
    Robust prediction:
     - image -> preprocess -> disease_model.predict -> treatability_model.predict
     - supports treatability models that output sigmoid (1) or softmax (2)
    Returns: (disease_name, treatability_status, solution)
    """
    # Load and preprocess image
    pil_img = open_image_force_rgb(img_path)
    x = preprocess_image_pil(pil_img, size=(224, 224))

    # Disease prediction
    disease_pred = disease_model.predict(x)
    if disease_pred is None or disease_pred.size == 0:
        raise ValueError("Empty disease prediction array")

    disease_probs = disease_pred[0] if disease_pred.ndim > 1 else disease_pred
    disease_index = int(np.argmax(disease_probs))
    if disease_index < 0 or disease_index >= len(disease_classes):
        raise ValueError(f"Predicted disease index {disease_index} out of range")

    disease_name = disease_classes[disease_index]

    # If healthy, skip treatability
    if disease_name in ["Healthy", "Pearl Healthy", "Sorghum (Jowar) Healthy"]:
        return disease_name, "N/A", disease_solutions.get(disease_name, "No solution available.")

    # Treatability prediction
    treat_pred = treatability_model.predict(x)
    if treat_pred is None or treat_pred.size == 0:
        raise ValueError("Empty treatability prediction array")

    treat_out = treat_pred[0] if treat_pred.ndim > 1 else treat_pred

    # If single neuron (sigmoid)
    if np.shape(treat_out)[-1] == 1:
        score = float(treat_out[0])
        treatability_status = "Treatable" if score > 0.5 else "Non-Treatable"
    # If two-class softmax
    elif np.shape(treat_out)[-1] == 2:
        idx = int(np.argmax(treat_out))
        treatability_status = "Treatable" if idx == 1 else "Non-Treatable"
    else:
        # Best-effort fallback
        idx = int(np.argmax(treat_out))
        treatability_status = "Treatable" if idx == 1 else "Non-Treatable"

    solution = disease_solutions.get(disease_name, "Solution not available.")
    return disease_name, treatability_status, solution

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles classic file upload (multipart/form-data).
    Renders result.html with prediction.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename_raw = secure_filename(file.filename)
        if not allowed_file_extension(filename_raw):
            return jsonify({"error": "Unsupported file format. Please upload JPG, PNG, or AVIF."}), 400

        ext = filename_raw.rsplit(".", 1)[-1].lower()
        unique_name = f"upload_{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)

        # If AVIF convert
        if ext == "avif":
            converted = convert_avif_to_jpg(filepath)
            if converted is None:
                os.remove(filepath)
                return jsonify({"error": "Failed to convert AVIF file."}), 400
            os.remove(filepath)
            filepath = converted
            unique_name = os.path.basename(filepath)

        # validate image
        try:
            with Image.open(filepath) as img:
                img.verify()
        except Exception:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Uploaded file is not a valid image."}), 400

        disease_name, treatability_status, solution = predict_disease(filepath)
        image_url = url_for("static", filename=f"uploads/{unique_name}")

        return render_template(
            "result.html",
            disease_name=disease_name,
            treatability_status=treatability_status,
            solution=solution,
            image_url=image_url
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"❌ Failed to process image: {str(e)}"}), 500

@app.route("/upload_camera", methods=["POST"])
def upload_camera():
    """
    Expects JSON: { "image_base64": "data:image/jpeg;base64,...." }
    Returns rendered result.html (HTML) so front-end can document.write it.
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        base64_image = data.get("image_base64") or data.get("image")
        if not base64_image:
            return jsonify({"error": "No image_base64 in JSON"}), 400

        # Strip header if present
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        image_bytes = base64.b64decode(base64_image)
        unique_filename = f"camera_{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Validate
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({"error": "Captured image is not a valid image"}), 400

        disease_name, treatability_status, solution = predict_disease(image_path)
        image_url = url_for("static", filename=f"uploads/{unique_filename}")

        return render_template(
            "result.html",
            disease_name=disease_name,
            treatability_status=treatability_status,
            solution=solution,
            image_url=image_url
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"❌ Failed to process image: {str(e)}"}), 500

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "GET":
        return render_template("chatbot.html")

    data = request.json or {}
    user_message = (data.get("message") or "").strip()

    if user_message in chatbot_answers:
        response_text = chatbot_answers[user_message]
        follow_up_questions = chatbot_questions.get(user_message, [])
    else:
        response_text = "I'm not sure. Please consult an agricultural expert."
        follow_up_questions = []

    return jsonify({"response": response_text, "questions": follow_up_questions})

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(debug=True)
