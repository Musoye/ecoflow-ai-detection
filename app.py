import os
import cv2
from flask import Flask, request, jsonify, send_file
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8n.pt',
    confidence_threshold=0.3,
    device="cpu"  # Change to "cuda" if you have an NVIDIA GPU
)
print("✅ Model Loaded!")

# --- ROUTES ---

@app.route('/health', methods=['GET'])
def health_check():
    """
    First request: Check if the API is up and running.
    """
    return jsonify({
        "status": "online",
        "message": "The SAHI Prediction API is up and running."
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Second request: Upload an image, predict, and return JSON result.
    """
    # 1. Check if image is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 2. Save the uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    try:
        # 3. Run Sliced Prediction (Your Logic)
        result = get_sliced_prediction(
            input_path,
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        # 4. Count People (Class ID 0)
        person_count = 0
        for object_prediction in result.object_prediction_list:
            if object_prediction.category.id == 0:
                person_count += 1

        # 5. Save Visualization
        # We save the visual output to the results folder to serve it back if needed
        output_filename = f"pred_{file.filename}"
        result.export_visuals(export_dir=RESULTS_FOLDER, file_name=output_filename)
        
        # Construct the full path where SAHI saved the image (SAHI adds extension)
        # Note: SAHI might save as png even if input was jpg, so we check the folder
        # For simplicity in this example, we just return the JSON data.
        
        return jsonify({
            "filename": file.filename,
            "estimated_people": person_count,
            "message": "Prediction successful"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5000)