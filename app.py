import os
import cv2
import PIL.Image
from flask import Flask, request, jsonify
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv() # Load environment variables from .env file

# 1. SETUP GEMINI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Fallback if .env fails (not recommended for production)
    GEMINI_API_KEY = "YOUR_ACTUAL_API_KEY_HERE" 

genai.configure(api_key=GEMINI_API_KEY)
# Use a model capable of vision (Gemini 1.5 Flash is fast and cheap)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# 2. SETUP FLASK
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 3. SETUP SAHI/YOLO (Global Load)
print("⏳ Loading YOLOv8 Model...")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8n.pt',
    confidence_threshold=0.3,
    device="cpu" 
)
print("✅ Model Loaded!")

def get_gemini_count(image_path):
    """
    Sends the image to Gemini with a prompt to count people.
    """
    try:
        img = PIL.Image.open(image_path)
        # Custom Prompt asking for a JSON-like single number response
        prompt = "Count the number of people in this image. Return ONLY the number as an integer. Do not write any text."
        
        response = gemini_model.generate_content([prompt, img])
        text_response = response.text.strip()
        
        # Clean up response in case Gemini is chatty
        # (e.g. removes "There are 5 people" -> "5")
        import re
        numbers = re.findall(r'\d+', text_response)
        if numbers:
            return int(numbers[0])
        return 1 # Avoid division by zero if it fails
    except Exception as e:
        print(f"Gemini Error: {e}")
        return 1 # Default to 1 to prevent crashes

# --- ROUTES ---

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "Hybrid SAHI + Gemini API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    try:
        # --- A. SAHI PREDICTION ---
        sahi_result = get_sliced_prediction(
            input_path,
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        sahi_count = 0
        for object_prediction in sahi_result.object_prediction_list:
            if object_prediction.category.id == 0: # Class 0 is Person
                sahi_count += 1

        # --- B. GEMINI PREDICTION ---
        gemini_count = get_gemini_count(input_path)

        # --- C. CALCULATION ---
        # "divide both then rounded to the nearest figure"
        # We handle division by zero just in case Gemini returns 0
        if gemini_count == 0:
            final_ratio = sahi_count # Or handle error differently
        else:
            raw_ratio = sahi_count / gemini_count
            final_ratio = round(raw_ratio)

        # Save visualization (Optional)
        output_filename = f"pred_{file.filename}"
        sahi_result.export_visuals(export_dir=RESULTS_FOLDER, file_name=output_filename)

        return jsonify({
            "filename": file.filename,
            "sahi_count": sahi_count,
            "gemini_count": gemini_count,
            "calculation_result": final_ratio,
            "formula": f"{sahi_count} / {gemini_count} rounded",
            "message": "Prediction successful"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Make sure to set your API Key in environment or .env file before running!
    app.run(debug=False, port=5000, host='0.0.0.0')