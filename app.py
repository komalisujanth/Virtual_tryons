from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load models once at startup
print("Loading AI models...")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✅ Models loaded")

skin_tone_map = {
    '1': (220, 200, 190),
    '2': (195, 170, 150),
    '3': (170, 130, 100),
    '4': (140, 100, 70),
    '5': (110, 75, 50),
    '6': (70, 45, 30),
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get files and params
        garment_file = request.files.get('garment')
        face_file = request.files.get('face')
        body_type = request.form.get('body_type', 'slim')
        height = request.form.get('height', 'medium')
        skin_tone = request.form.get('skin_tone', '3')

        if not garment_file or not face_file:
            return jsonify({'success': False, 'error': 'Missing files'})

        # Save uploaded files
        uid = str(uuid.uuid4())[:8]
        garment_path = os.path.join(UPLOAD_FOLDER, f'{uid}_garment.jpg')
        face_path = os.path.join(UPLOAD_FOLDER, f'{uid}_face.jpg')
        garment_file.save(garment_path)
        face_file.save(face_path)

        # Load garment image
        garment_img = cv2.imread(garment_path)
        if garment_img is None:
            return jsonify({'success': False, 'error': 'Cannot load garment image'})

        # Check if body model exists, otherwise use garment image directly
        model_path = f'models/{body_type}_{height}.jpg'
        if os.path.exists(model_path):
            base_img = cv2.imread(model_path)
        else:
            # Fall back to garment image if that body type combo not available
            base_img = garment_img.copy()

        # Load user face
        user_img = cv2.imread(face_path)
        if user_img is None:
            return jsonify({'success': False, 'error': 'Cannot load face image'})

        # Detect faces
        base_faces = face_app.get(base_img)
        user_faces = face_app.get(user_img)

        if len(base_faces) == 0:
            # Try garment image as fallback
            base_img = garment_img.copy()
            base_faces = face_app.get(base_img)

        if len(base_faces) == 0:
            return jsonify({'success': False, 'error': 'No face detected in model image. Try a different body type or upload a clearer garment photo.'})

        if len(user_faces) == 0:
            return jsonify({'success': False, 'error': 'No face detected in your photo. Please retake.'})

        # Swap face
        result = base_img.copy()
        result = swapper.get(result, base_faces[0], user_faces[0], paste_back=True)

        # Apply skin tone to exposed skin areas
        skin_color = skin_tone_map.get(skin_tone, (140, 100, 70))
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 60])
        upper = np.array([25, 180, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.dilate(mask, None, iterations=2)
        colored = result.copy()
        colored[mask > 0] = skin_color
        result = cv2.addWeighted(result, 0.65, colored, 0.35, 0)

        # Save result
        result_filename = f'result_{uid}.jpg'
        result_path = os.path.join(OUTPUT_FOLDER, result_filename)
        cv2.imwrite(result_path, result)

        # Cleanup uploads
        os.remove(garment_path)
        os.remove(face_path)

        return jsonify({
            'success': True,
            'result_url': f'/output/{result_filename}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 VirtualFit running at http://localhost:5000")
    app.run(debug=False, port=5000)