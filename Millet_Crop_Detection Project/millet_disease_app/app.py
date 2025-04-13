import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
import base64
import uuid
import pillow_avif

app = Flask(__name__)

# Ensure "uploads" folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define absolute paths to the model files
disease_model_path = os.path.abspath(
    r"C:\Users\91843\Desktop\Projects\Millet_Crop_Detection Project\millet_disease_app\models\MobileNetV2_millet_model.h5"
)
treatability_model_path = os.path.abspath(
    r"C:\Users\91843\Desktop\Projects\Millet_Crop_Detection Project\millet_disease_app\models\stage_classification_model.h5"
)

# Check if models exist
if not os.path.exists(disease_model_path):
    raise FileNotFoundError(f"❌ ERROR: Disease model not found at {disease_model_path}")
if not os.path.exists(treatability_model_path):
    raise FileNotFoundError(f"❌ ERROR: Treatability model not found at {treatability_model_path}")

# Load models
disease_model = load_model(disease_model_path)
treatability_model = load_model(treatability_model_path)

# Class labels for disease classification
disease_classes = [
    "Finger (Ragi) Downy", "Finger (Ragi) Mottle", "Finger (Ragi) Smut", "Finger (Ragi) Wilt",
    "Healthy", "Pearl Healthy", "Pearl Rust Disease", "Pearl Downy Mildew",
    "Sorghum (Jowar) Blast", "Sorghum (Jowar) Ergot", "Sorghum (Jowar) Smut",
    "Sorghum (Jowar) Rust", "Sorghum (Jowar) Healthy"
]

# Solutions for each disease
disease_solutions = {
    "Finger (Ragi) Downy": "Spray Metalaxyl 8% + Mancozeb 64% WP at 2 g/L water...",
    "Finger (Ragi) Mottle": "Avoid excessive nitrogen fertilizers...",
    "Finger (Ragi) Smut": "Treat seeds with Captan or Thiram @ 2g/kg of seed...",
    "Finger (Ragi) Wilt": "Improve soil drainage to reduce fungal buildup...",
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

# Image preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input size
    img = np.array(img) / 255.0   # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model
    return img

# Prediction function
def predict_disease(img_path):
    img = Image.open(img_path)
    img_array = preprocess_image(img)

    # Predict disease
    disease_prediction = disease_model.predict(img_array)
    disease_index = np.argmax(disease_prediction)
    disease_name = disease_classes[disease_index]

    # If the disease is "Healthy", no need to check for treatability
    if disease_name in ["Healthy", "Pearl Healthy", "Sorghum (Jowar) Healthy"]:
        return disease_name, "N/A", disease_solutions[disease_name]

    # Predict treatability
    treatability_prediction = treatability_model.predict(img_array)
    treatability_status = "Treatable" if treatability_prediction[0][0] > 0.5 else "Non-Treatable"

    # Get the solution based on the disease name
    solution = disease_solutions.get(disease_name, "Solution not available.")

    return disease_name, treatability_status, solution

def convert_avif_to_jpg(avif_path):
    """Converts an AVIF image to JPEG format using Pillow (PIL)."""
    try:
        img = Image.open(avif_path)
        jpg_path = avif_path.replace(".avif", ".jpg")
        img.save(jpg_path, "JPEG")
        print(f"✅ AVIF converted successfully: {jpg_path}")
        return jpg_path
    except Exception as e:
        print(f"❌ Error in AVIF conversion: {e}")
        return None  # Return None if conversion fails

    
# Predefined chatbot flow
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

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    # Get file extension
    file_ext = file.filename.split(".")[-1].lower()

    # Validate supported formats
    if file_ext not in ["jpg", "jpeg", "png", "avif"]:
        return jsonify({"error": "Unsupported file format. Please upload JPG, PNG, or AVIF."})

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Convert AVIF to JPG (if needed)
    if file_ext == "avif":
        converted_path = convert_avif_to_jpg(filepath)

        if converted_path is None:
            return jsonify({"error": "⚠️ Failed to process AVIF file. Please upload a valid image."}), 400

        filepath = converted_path  # Use the converted JPG file

    # Try to open the image with PIL
    try:
        img = Image.open(filepath)
    except UnidentifiedImageError:
        return jsonify({"error": "Unable to process image. Ensure it's a valid image file."})

    # ✅ Predict disease & treatability
    disease_name, treatability_status, solution = predict_disease(filepath)

    # ✅ Fix image path for UI
    image_url = "/" + filepath.replace("\\", "/")  # Fix Windows path issues for static files

    # ✅ Render result page with prediction data
    return render_template(
        "result.html",
        disease_name=disease_name,
        treatability_status=treatability_status,
        solution=solution,
        image_url=image_url
    )


@app.route("/camera")
def camera():
    return render_template("camera.html")

@app.route("/upload_camera", methods=["POST"])
def upload_camera():
    """Handles image upload from the camera (Base64 conversion)."""
    data = request.json
    base64_image = data.get("image_base64")

    if not base64_image:
        return jsonify({"error": "⚠️ No image data received"}), 400

    try:
        # 🔍 Ensure Base64 format is correct
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]  # Remove "data:image/jpeg;base64," part

        # Decode Base64 image
        image_data = base64.b64decode(base64_image)

        # Save image
        unique_filename = f"camera_{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        with open(image_path, "wb") as f:
            f.write(image_data)

        # 🔍 Verify image file integrity
        try:
            with Image.open(image_path) as img:
                img.verify()  # Checks if image is corrupted
        except Exception:
            os.remove(image_path)  # Delete invalid file
            return jsonify({"error": "⚠️ Captured image is not a valid JPEG/PNG"}), 400

        # 🔍 Predict disease using ML Model
        disease_name, treatability_status, solution = predict_disease(image_path)

        # ✅ Return Redirect URL
        return jsonify({
            "redirect_url": url_for("result", disease_name=disease_name, treatability_status=treatability_status, solution=solution, image_url="/" + image_path.replace("\\", "/"))
        })
    except Exception as e:
        return jsonify({"error": f"❌ Failed to process image: {str(e)}"}), 500  

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    if request.method == "GET":
        return render_template("chatbot.html")

    data = request.json
    user_message = data.get("message", "").strip()

    if user_message in chatbot_answers:
        response_text = chatbot_answers[user_message]
        follow_up_questions = chatbot_questions.get(user_message, [])
    else:
        response_text = "I'm not sure. Please consult an agricultural expert."
        follow_up_questions = []

    return jsonify({"response": response_text, "questions": follow_up_questions})

if __name__ == "__main__":
    app.run(debug=True)
