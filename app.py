# Import necessary Flask components and libraries for AI model integration
import os
import threading

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Inference runs on LiteRT (TFLite), not full TensorFlow. TensorFlow needs
# ~600 MB of RAM just to import, which does not fit the 512 MB free tier this
# app is deployed on; the converted .tflite model runs the same network in
# ~150 MB. Three import paths are tried so the same file works everywhere:
#   ai_edge_litert  -- the maintained runtime, used in production (Linux)
#   tflite_runtime  -- older name, still found on some systems
#   tensorflow.lite -- fallback for local development, where only TF is present
#                      (ai-edge-litert publishes no Windows wheels)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # pragma: no cover
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        # Note the attribute access: tf.lite is a lazily-populated API
        # namespace, not the tensorflow.lite package, so
        # `from tensorflow.lite import Interpreter` raises ImportError here.
        import tensorflow as _tf
        Interpreter = _tf.lite.Interpreter

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS to allow the frontend to communicate with the backend
CORS(app)

# --- Model Loading and Configuration ---
# The converted model ships alongside this file, so resolve it relative to this
# module rather than to the current working directory -- that way the app runs
# from any directory and on any machine. Set the MODEL_PATH environment
# variable to load a different model file instead.
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eye_disease_model.tflite')
)

# The size the network was trained on. Uploads are resized to this.
INPUT_SIZE = (256, 256)

# Define the class labels your model was trained to predict.
# The order of these labels MUST match the order of your model's output,
# which is determined by the alphabetical order of the directory names.
CLASS_LABELS = [
    'Cataract',
    'Diabetic Retinopathy',
    'Glaucoma',
    'Normal'
]

# Globals holding the loaded interpreter and its tensor metadata
interpreter = None
input_detail = None
output_detail = None

# A TFLite interpreter holds mutable internal buffers, so two requests calling
# invoke() at once would corrupt each other's results. gunicorn runs this with
# several threads to stay responsive, so serialise the inference itself.
inference_lock = threading.Lock()

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
    """Loads the converted TFLite model and caches its tensor metadata."""
    global interpreter, input_detail, output_detail
    try:
        # num_threads=1 because the free tier allocates a fraction of a CPU;
        # extra threads only add contention there.
        interpreter = Interpreter(model_path=MODEL_PATH, num_threads=1)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        print(f"Model loaded successfully! input={input_detail['shape']} "
              f"dtype={np.dtype(input_detail['dtype']).name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        interpreter = None


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
    if interpreter is None:
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
        image = image.resize(INPUT_SIZE)
        # EfficientNetV2 takes pixels in the [0, 255] range: its rescaling is
        # baked into the network, which is why keras' efficientnet_v2
        # preprocess_input is a no-op. tools/convert_to_tflite.py asserts this,
        # so there is nothing to reproduce here beyond the dtype change.
        image_array = np.asarray(image, dtype=np.float32)
        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Make a prediction with the model
        with inference_lock:
            interpreter.set_tensor(input_detail['index'],
                                   image_array.astype(input_detail['dtype']))
            interpreter.invoke()
            # Copy the result out before releasing the lock -- the buffer is
            # reused by the next invoke().
            predictions = np.array(interpreter.get_tensor(output_detail['index']))

        # The network's final Dense layer is linear, so these are raw logits,
        # not probabilities -- they go negative and do not sum to 1. argmax is
        # unaffected by that, but reporting a confidence means applying softmax
        # explicitly. Subtracting the max first stops exp() from overflowing.
        logits = np.asarray(predictions[0], dtype=np.float64)
        exp_logits = np.exp(logits - logits.max())
        probabilities = exp_logits / exp_logits.sum()

        predicted_class_index = int(np.argmax(logits))
        predicted_label = CLASS_LABELS[predicted_class_index]
        confidence = float(probabilities[predicted_class_index])

        # Get the appropriate response from the dictionary. Copy it, so that
        # adding 'diagnosis' does not mutate the shared DIAGNOSIS_INFO entry.
        if predicted_label in DIAGNOSIS_INFO:
            response_data = dict(DIAGNOSIS_INFO[predicted_label])
            response_data['diagnosis'] = predicted_label
            response_data['confidence'] = round(confidence, 4)
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
    # Debug mode restarts the server on code changes, but it also exposes an
    # interactive Python console on any traceback -- never leave it enabled on
    # a public deployment. Off unless FLASK_DEBUG is set for local development.
    debug = os.environ.get('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
    # PORT is supplied by the host in production; 5000 keeps local runs familiar.
    app.run(debug=debug, port=int(os.environ.get('PORT', 5000)))
