from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps
import os

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
    string2 = request.form.get('string2')
    float1 = float(request.form.get('float1'))
    float2 = float(request.form.get('float2'))
    float3 = float(request.form.get('float3'))
    
    # Print the received variables
    print(f"Received string1: {string1}")
    print(f"Received string2: {string2}")
    print(f"Received float1: {float1}")
    print(f"Received float2: {float2}")
    print(f"Received float3: {float3}")

    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400
    if image:
        # Ensure filename has an extension
        filename = image.filename
        if '.' not in filename:
            return jsonify({"error": "No file extension"}), 400
        
        # Save original image
        original_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(original_path)
        
        # Open and process image to grayscale
        img = Image.open(original_path)
        fractal_generator = gpuNewt.NewtonFractalGenerator(width=800, height=800, image_path=original_path)
        img = fractal_generator.generate_image("z*z*z-1", 1)
        processed_path = os.path.join(PROCESSED_FOLDER, filename)
        img.save(processed_path)
        
        return jsonify({"message": "Image uploaded and processed successfully", "processed_path": f"/processed/{filename}"}), 200

@app.route('/processed/<filename>')
def processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
