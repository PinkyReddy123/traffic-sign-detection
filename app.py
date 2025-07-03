from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import torch

app = Flask(__name__)

# ====== CONFIGURATION ======
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
cnn_model = load_model('gtsrb_cnn_model.h5')
print("CNN model loaded successfully")


# ====== LOAD MODELS ======
cnn_model = load_model('gtsrb_cnn_model.h5')
yolo_model = torch.hub.load('yolov5', 'custom', path='yolov5/best.pt', source='local')

# ====== ROUTES ======
@app.route('/')
def index():
    return render_template('index.html')


def preprocess_for_cnn(img_path):
    img = Image.open(img_path).resize((64, 64))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded!', 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # ----- CNN Prediction -----
    cnn_input = preprocess_for_cnn(image_path)
    cnn_pred = cnn_model.predict(cnn_input)
    class_id = np.argmax(cnn_pred)
    confidence = np.max(cnn_pred)

    # ----- YOLO Detection -----
    results = yolo_model(image_path)
    results.render()
    result_image = results.ims[0]
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    cv2.imwrite(result_path, result_image_bgr)

    return f"""
    <h2>Prediction Complete!</h2>
    <p><b>CNN Classification:</b> Class {class_id} with {confidence:.2f} confidence</p>
    <h3>YOLO Detection:</h3>
    <img src="/{result_path}" width="400">
    <br><br><a href="/">Try Another</a>
    """


# ====== RUN APP ======
if __name__ == '__main__':
    app.run(debug=True)

