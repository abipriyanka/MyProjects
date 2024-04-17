# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 19:43:43 2023

@author: ABI PRIYANKA
"""


from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from sklearn.metrics import confusion_matrix

transformers = {
    'train_transforms': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test_transforms': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid_transforms': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

trans = list(transformers.keys())
path = 'C:/Users/ABI PRIYANKA/Downloads/pnuemonia'
categories = ['train', 'valid', 'test']

dset = {
    x: datasets.ImageFolder(
        os.path.join(path, x),
        transform=transformers[trans[i]]
    )
    for i, x in enumerate(categories)
}

class_names = dset['train'].classes
num_threads = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a new instance of the ResNet model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Load the saved model state dictionary
saved_model_filename = 'good_model.pth'
model.load_state_dict(torch.load(saved_model_filename))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Create a test dataloader
test_dataloader = torch.utils.data.DataLoader(
    dset['test'], batch_size=1, shuffle=False, num_workers=num_threads
)

# Initialize lists to store predictions and ground truth labels
predictions = []
labels = []

# Iterate over test data
for inputs, targets in test_dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    # Store the predicted and actual labels
    predictions.append(preds.item())
    labels.append(targets.item())

# Calculate the accuracy
accuracy = (np.array(predictions) == np.array(labels)).mean()
print("Accuracy:", accuracy)




from PIL import Image
import torch
import torchvision.transforms as transforms

# Load the image using PIL
test_image = Image.open(
    r"C:\Users\ABI PRIYANKA\Downloads\pnuemonia\test\PNEUMONIA\person103_bacteria_490.jpeg"
)


# Convert the image to RGB if it's grayscale
if test_image.mode != "RGB":
    test_image = test_image.convert("RGB")
    
    
# Define the transformation to apply to the image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the transformation to the image
test_image = transform(test_image)

# Add a batch dimension to match the expected input shape of the model
test_image = test_image.unsqueeze(0)

# Move the image tensor to the device (CPU or GPU)
test_image = test_image.to(device)

# Use the trained model to make a prediction on the input image
result = model(test_image)

# Convert the result tensor to probabilities using softmax
probabilities = torch.softmax(result, dim=1)

# Extract the probability of the input image belonging
# to each class from the prediction result
class_probabilities = probabilities[0]

# Determine the class with the highest probability and print its label
if class_probabilities[0] > class_probabilities[1]:
    print("Normal")
else:
    print("Pneumonia")






