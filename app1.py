import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from utils import encode_img, decode_img, sigmoid, post_process_mask, get_transforms
import onnxruntime as ort

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL = ort.InferenceSession("models/dresunet.onnx")
transform = get_transforms()

def process_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))  # Resize the image if necessary
    image = image.astype(np.float32) / 255.0  # Normalize the image to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Change the shape to (channels, height, width)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform segmentation
    onix_input = {"in_image": image}
    mask, prob = MODEL.run(None, onix_input)
    mask, prob = sigmoid(mask[0][0]), sigmoid(prob[0][0])
    mask = post_process_mask(mask, prob, mask_thresh=0.5, min_area=2500, pred_thres=0.5)
    mask = encode_img(mask)

    # Ensure mask has a fixed data type
    mask = np.array(mask, dtype=bool)

    # Check if there are any segmented regions
    has_segmented_regions = np.sum(mask) > 0

    # Set the message based on the presence of segmented regions
    message = "Pneumothorax is Present" if has_segmented_regions else "Pneumothorax is not present"

    return mask, message

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        mask, message = process_image(file_path)

        # Update the mask path to serve from the uploads folder
        mask_path = f"/uploads/{filename}"
        return render_template('result.html', mask=mask_path, message=message)

    return jsonify({'error': 'Unknown error'})


if __name__ == '__main__':
    app.run(debug=True)

