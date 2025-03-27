from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder="templates", static_folder="static")

# Fixed class order to match model output
class_names = [
    "Hypertension, Cardiovascular Disease", 
    "Down Syndrome - Heart Defects and Immune Issues", 
    "Neurological Disorders - Autism, Schizophrenia",
    "Scleroderma - Tissue Disorder",
    "Healthy"
]

# Load trained model
model = load_model("C:/all pros/palm pro/palm disease app/model/palm_disease_model.h5")
input_shape = model.input_shape[1:]

# Upload directory for storing images
UPLOAD_FOLDER = os.path.join('static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(img_path):
    """Preprocess image dynamically to match model input."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (input_shape[0], input_shape[1]))  # Resize dynamically
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    img = img.astype("float32") / 255.0  # Normalize pixel values
    return img

def predict_image(img_path):
    """Predict disease based on palm image."""
    img = preprocess_image(img_path)
    prediction = model.predict(img)[0]  # Get model prediction
    predicted_class = np.argmax(prediction)  # Get class with highest probability
    confidence = prediction[predicted_class] * 100  # Confidence in percentage
    return class_names[predicted_class], confidence

def generate_processed_image(img_path):
    """Generates palm-line image with black background and white palm lines."""
    img = cv2.imread(img_path)
    if img is None:
        return None  # Return None if image cannot be read

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform Canny edge detection
    edges = cv2.Canny(gray_img, 50, 150)

    # Create a black background
    black_background = np.zeros_like(img)

    # Highlight the edges (white) on the black background
    black_background[edges == 255] = (255, 255, 255)  # White color for edges

    return black_background  # Return the image with highlighted palm lines

@app.route("/", methods=["GET"])
def index():
    """Render main page."""
    return render_template("index.html")

# Modify the predict function to ensure the confidence is formatted to two decimal places
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    image_path = None
    highlighted_base64 = None

    if "image" in request.files:
        image = request.files["image"]
        if image:
            filename = secure_filename(image.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(img_path)

            # Generate highlighted image with black background and white palm lines
            highlighted = generate_processed_image(img_path)
            if highlighted is not None:
                _, highlighted_encoded = cv2.imencode('.png', highlighted)
                highlighted_base64 = base64.b64encode(highlighted_encoded).decode('utf-8')

            # Get prediction & confidence
            prediction, confidence = predict_image(img_path)

            # Format confidence to two decimal places
            confidence = "{:.2f}".format(confidence)

            # Convert image path to a relative URL
            image_path = f"/static/images/{filename}"

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        highlighted_base64=highlighted_base64
    )

if __name__ == "__main__":
    app.run(debug=True)
