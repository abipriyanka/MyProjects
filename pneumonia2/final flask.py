from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision import models

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('good_model.pth', map_location=device))
model = model.to(device)
model.eval()


# Define the transformation for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Handle the predict route
@app.route('/predict', methods=['POST'])
def predict():
    
    print(request.files)

    # Get the uploaded image from the request
    uploaded_file = request.files['img']
    image = Image.open(uploaded_file).convert('RGB')
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make predictions using the preprocessed image
    with torch.no_grad():
        outputs = model(preprocessed_image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    
    # Process the predictions and return the response
    if predicted_class == 0:
        result = "Normal"
    else:
        result = "Pneumonia"
    
    return render_template('result.html', result=result)

# Handle the home route
@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.debug=False
    app.run(port=8073)
