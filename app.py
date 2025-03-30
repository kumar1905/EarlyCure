from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
import imghdr

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
model = load_model("C:\\Users\\madas\\OneDrive\\Desktop\\PROJECTS\\EarlyCure\\Early_Cure_WebApp\\api\\model\\palm_disease_model.h5")
input_shape = model.input_shape[1:]

def preprocess_image_from_memory(img):
    """Preprocess image dynamically to match model input."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_resized = cv2.resize(img_gray, (input_shape[0], input_shape[1]))  # Resize
    img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    img_resized = img_resized.astype("float32") / 255.0  # Normalize
    return img_resized

def predict_from_memory(img):
    """Predict disease based on palm image in memory."""
    prediction = model.predict(img)[0]  # Get model prediction
    predicted_class = np.argmax(prediction)  # Get class with highest probability
    confidence = prediction[predicted_class] * 100  # Confidence in percentage
    return class_names[predicted_class], confidence

def generate_processed_image_from_memory(img):
    """Generates palm-line image with black background and white palm lines."""
    try:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Apply histogram equalization to enhance contrast
        gray_img = cv2.equalizeHist(gray_img)

        # Adaptive Thresholding to enhance palm lines
        thresh = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )

        # Dilation to make lines thicker
        kernel = np.ones((3, 3), np.uint8)
        processed_img = cv2.dilate(thresh, kernel, iterations=1)

        return processed_img

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route("/", methods=["GET"])
def index():
    """Render main page."""
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    uploaded_image_base64 = None
    highlighted_base64 = None
    mime_type = "image/jpeg"  # Default MIME type

    if request.method == "POST" and "image" in request.files:
        image = request.files["image"]
        if image.filename != '':
            try:
                img_bytes = image.read()
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img is not None:
                    uploaded_image_base64 = base64.b64encode(img_bytes).decode('utf-8')

                    # Detect MIME type using imghdr
                    image_type = imghdr.what(None, h=img_bytes)
                    if image_type:
                        mime_type = f"image/{image_type}"
                    else:
                        filename = image.filename.lower()
                        if filename.endswith(('.png', '.jpeg', '.jpg', '.gif', '.bmp', '.tiff', '.tif', '.webp')):
                            ext = filename.rsplit('.', 1)[1]
                            mime_type = f"image/{ext}"
                        else:
                            mime_type = "image/jpeg"

                    # Generate palm markings image
                    highlighted = generate_processed_image_from_memory(img)
                    if highlighted is not None:
                        _, highlighted_encoded = cv2.imencode('.png', highlighted)
                        highlighted_base64 = base64.b64encode(highlighted_encoded).decode('utf-8')

                    # Preprocess and predict
                    processed_img = preprocess_image_from_memory(img)
                    prediction, confidence = predict_from_memory(processed_img)
                    confidence = "{:.2f}".format(float(confidence))

                    

            except Exception as e:
                print(f"Error: {str(e)}")
                return render_template("predict.html", error=str(e))

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        uploaded_image_base64=uploaded_image_base64,
        highlighted_base64=highlighted_base64,
        mime_type=mime_type,
        predicted_disease=prediction # Pass prediction to the template
    )

if __name__ == "__main__":
    app.run(debug=True)