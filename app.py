import os
import secrets
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Try to import TensorFlow/keras. We wrap this in a try-except block so the Flask app
# still runs even if tensorflow isn't installed yet, allowing us to see the UI.
try:
    from tensorflow.keras.models import load_model as keras_load_model
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow is not installed. Model predictions will not work.")

# Initialize Flask App
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'images')
app.config['STATIC_FOLDER'] = 'static'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None

# --- Model Configuration ---
# You may need to adjust these according to how your model was trained!
IMAGE_SIZE = (224, 224) 
CLASS_LABELS = {
    0: "Healthy",
    1: "Disease Type 1",
    2: "Disease Type 2",
    # TODO: Add your actual class names here corresponding to the model's output neurons
}

def load_ai_model():
    """Loads the pre-trained Keras model from the models directory."""
    global model
    if not TF_AVAILABLE:
        print("Skipping AI model load: TensorFlow not available.")
        return

    model_path = os.path.join('model', 'plant_disease_model_final.h5')
    if os.path.exists(model_path):
        try:
            model = keras_load_model(model_path, compile=False)
            print("AI Model loaded successfully.")
        except Exception as e:
            print(f"Error loading AI model: {e}")
    else:
        print(f"Error: Model file not found at {model_path}")

def predict_image(filepath):
    """Processes an image and returns the model's prediction."""
    if not TF_AVAILABLE or model is None:
        return "Model not loaded (TensorFlow missing or model file not found)"

    try:
        # Load and preprocess the image
        img = keras_image.load_img(filepath, target_size=IMAGE_SIZE)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100
        
        # Get label (fallback to index if not in dictionary)
        label = CLASS_LABELS.get(predicted_class_idx, f"Class {predicted_class_idx}")
        return f"{label} ({confidence:.2f}% confidence)"
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error making prediction"

# --- Routes ---

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 1. Get prediction from AI model
            prediction = predict_image(filepath)

            # 2. Save image permanently to static folder for Displaying
            static_filename = f"{secrets.token_hex(8)}_{filename}"
            static_path = os.path.join(app.config['STATIC_FOLDER'], 'images', static_filename)
            
            # Use PIL to verify it's a valid image and save it
            image = Image.open(filepath)
            # convert to RGB if it's RGBA (for PNGs)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(static_path)

            # 3. Store results in Flask session so they can be accessed on the /result page
            session['prediction'] = prediction
            session['image_path'] = f"images/{static_filename}"

            # 4. Clean up the original uploaded temp file
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({'success': True})

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500


@app.route('/result')
def result():
    prediction = session.get('prediction')
    image_path = session.get('image_path')

    if not prediction:
        return redirect(url_for('upload'))

    return render_template('result.html', prediction=prediction, image_path=image_path)


if __name__ == '__main__':
    # Load AI model into memory before starting server
    load_ai_model()
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)