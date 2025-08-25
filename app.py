# Import necessary Flask components and libraries for AI model integration
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Import the specific preprocessing function for EfficientNetV2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess_input

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS to allow the frontend to communicate with the backend
CORS(app)

# --- Model Loading and Configuration ---
# Set the path to your pre-trained Keras model file
# REPLACE 'your_model.keras' with the actual filename of your model
MODEL_PATH = 'C:/Users/hp/Desktop/olaprojects/eye_disease_model.keras'

# Define the class labels your model was trained to predict.
# The order of these labels MUST match the order of your model's output,
# which is determined by the alphabetical order of the directory names.
CLASS_LABELS = [
    'Cataract',
    'Diabetic Retinopathy',
    'Glaucoma',
    'Normal'
]

# A global variable to hold the loaded model
model = None

# A dictionary to store detailed responses for each diagnosis
DIAGNOSIS_INFO = {
    'Normal': {
        'message': 'No significant findings detected. Your eyes appear healthy.',
        'recommendations': [
            'Maintain a healthy diet and lifestyle.',
            'Schedule regular comprehensive eye exams.',
            'Protect your eyes from UV rays.'
        ]
    },
    'Diabetic Retinopathy': {
        'message': 'There are signs of diabetic retinopathy. Immediate consultation with a specialist is highly recommended.',
        'recommendations': [
            'Schedule an appointment with an ophthalmologist.',
            'Monitor blood sugar levels closely.',
            'Follow a balanced diet and regular exercise routine.'
        ]
        
    },
    'Glaucoma': {
        'message': 'Potential signs of glaucoma were detected. Consult an eye care professional for further evaluation.',
        'recommendations': [
            'See an eye doctor for a full eye pressure test.',
            'Do not miss scheduled follow-up appointments.',
            'Inform family members, as glaucoma can be hereditary.'
        ]
    },
    'Cataract': {
        'message': 'Possible signs of cataracts were found. It is recommended to see an ophthalmologist for a definitive diagnosis.',
        'recommendations': [
            'Consult with an eye care specialist to discuss treatment options.',
            'Wear sunglasses to protect your eyes from further damage.'
        ]
    }
}

# Load the model when the application starts
def load_model():
    """Loads the pre-trained Keras model from the file."""
    global model
    try:
        # Load the model from the specified path
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Register the model loading function to run before the first request
with app.app_context():
    load_model()


# The main page route will serve the HTML content from the templates folder
@app.route('/')
def serve_html():
    """Serves the main HTML page for the application."""
    print("Serving the main HTML page.")
    # render_template looks for a file named index.html in the 'templates' folder
    return render_template('index.html')

# The prediction route will handle the image analysis POST request
@app.route('/predict', methods=['POST'])
def predict():
    """Handles image uploads and returns an analysis result from the AI model."""
    global model

    if model is None:
        return jsonify({
            "diagnosis": "Model Not Loaded",
            "message": "The AI model could not be loaded. Please check the model file path.",
            "recommendations": []
        }), 500

    # Check if a file was uploaded in the request
    if 'file' not in request.files:
        return jsonify({
            "diagnosis": "No file uploaded",
            "message": "Please upload a retinal image.",
            "recommendations": []
        }), 400

    file = request.files['file']

    try:
        # Preprocess the image for the model
        image = Image.open(file.stream).convert("RGB")
        # Resize the image to match the model's expected input shape
        image = image.resize((256, 256))
        image_array = np.array(image)
        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        # Use the specific EfficientNet preprocessing function
        image_array = efficientnet_preprocess_input(image_array)

        # Make a prediction with the model
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions)
        predicted_label = CLASS_LABELS[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])

        # Get the appropriate response from the dictionary
        if predicted_label in DIAGNOSIS_INFO:
            response_data = DIAGNOSIS_INFO[predicted_label]
            response_data['diagnosis'] = predicted_label
        else:
            response_data = {
                'diagnosis': 'Unknown Result',
                "message": 'The model returned an unexpected result.',
                "recommendations": []
            }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            "diagnosis": "Analysis Failed",
            "message": f"An error occurred during analysis: {str(e)}",
            "recommendations": []
        }), 500

# Run the application
if __name__ == '__main__':
    # Setting debug to True will automatically restart the server on code changes
    # It should be set to False in a production environment
    app.run(debug=True)
