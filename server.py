
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import os
import time

import gpuNewt

app = Flask(__name__, static_folder="processed_images")
CORS(app)
UPLOAD_FOLDER = "flask_image_uploads"
PROCESSED_FOLDER = "processed_images"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    image = request.files['image']
    string1 = request.form.get('string1')
    
    # Print the received variables for debugging
    print(f"Received string1: {string1}")

    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400
    if image:
        # Ensure filename has an extension
        filename, ext = os.path.splitext(image.filename)
        if not ext:
            return jsonify({"error": "No file extension"}), 400
        
        # Attach a Unix timestamp to the filename
        timestamp = str(int(time.time()))
        new_filename = f"{filename}_{timestamp}{ext}"
        
        # Save original image
        original_path = os.path.join(UPLOAD_FOLDER, new_filename)
        image.save(original_path)
        
        # Open and process image to grayscale
        img = Image.open(original_path)
        fractal_generator = gpuNewt.NewtonFractalGenerator(width=1920, height=1920, image_path=original_path)
        img = fractal_generator.generate_image(string1, 1)
        processed_path = os.path.join(PROCESSED_FOLDER, new_filename)
        img.save(processed_path)
        
        return jsonify({"message": "Image uploaded and processed successfully", "processed_path": f"/processed/{new_filename}"}), 200

@app.route('/processed/<filename>')
def processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
