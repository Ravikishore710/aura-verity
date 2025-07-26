# app.py (Corrected Version with CORS)
# Main Flask application file.

import os
from flask import Flask, request, jsonify, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model_loader import load_prediction_model
from processing import run_analysis_pipeline

# --- Configuration ---
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# --- App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'supersecretkey_for_real'

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Loading ---
print("Loading model... This may take a moment.")
try:
    model = load_prediction_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load the model. Error: {e}")
    model = None

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Route for Analysis ---
@app.route('/analyze', methods=['POST'])
def analyze_file():
    """Handles the file upload and analysis, returning JSON."""
    if model is None:
        return jsonify({"error": "Model is not available. Please check server logs."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run the full analysis pipeline
        try:
            analysis_results = run_analysis_pipeline(filepath, model)
            return jsonify(analysis_results)
        except Exception as e:
            print(f"Error during analysis pipeline: {e}")
            return jsonify({"error": f"An unexpected error occurred during analysis: {e}"}), 500

    else:
        return jsonify({"error": "Invalid file type."}), 400

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)
