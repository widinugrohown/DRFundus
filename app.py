from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
from torchvision.models import efficientnet_b0

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

num_classes = 5

model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Load the saved weights
model.load_state_dict(torch.load(r'model\best_efficientnet_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = ['0', '1', '2', '3', '4']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]
            prediction_probability = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()

        return jsonify({
            'class_name': predicted_class,
            'prediction_probability': prediction_probability,
            'image_path': url_for('static', filename=f'uploads/{filename}')
        })

if __name__ == '__main__':
    app.run(debug=True)
