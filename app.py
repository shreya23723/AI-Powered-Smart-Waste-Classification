from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("saved_model/garbage_classifier.h5")

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# API Route for Image Classification
@app.route("/classify", methods=["POST"])
def classify():
    if not request.data:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = "static/uploads/temp_image.jpg"  # Save in static folder
    with open(file_path, "wb") as f:
        f.write(request.data)

    # Preprocess and predict
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    result = "Recyclable" if prediction[0][0] > 0.5 else "Organic"

    return result  # Send classification result as plain text

# Run Flask server
if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)  # Ensure upload folder exists
    app.run(debug=True)
