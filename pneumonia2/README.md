# Pneumonia Detection Model

## Introduction:
Pneumonia is a significant respiratory infection with high morbidity and mortality rates. Early and accurate detection is crucial for timely treatment and improved patient outcomes. Automated pneumonia detection models offer a solution to the challenges associated with manual interpretation of chest X-ray images.

## Dataset:
The dataset comprises chest X-ray images from pediatric patients, sourced from Guangzhou Women and Children’s Medical Centre. Images were graded by expert physicians, ensuring quality and accuracy for training the AI system.

## Environment and Tools:
- Scikit-learn
- PyTorch
- NumPy
- Pandas
- Matplotlib
- PIL
- Flask

## Data Preprocessing:
Data augmentation and class balancing techniques were employed to enhance model performance. These include resizing images, random rotations, horizontal flipping, and up/down sampling for class balance.

## Model Architecture:
The model architecture utilizes the ResNet-18 CNN architecture, fine-tuned for pneumonia detection. ResNet-18 is a deep neural network with residual connections, allowing for effective feature extraction from chest X-ray images.

## Training:
Training involves stochastic gradient descent optimization, forward pass, loss computation, backpropagation, and parameter updates. The model was trained over five epochs, achieving high training and validation accuracy.

## Evaluation Metrics:
Evaluation metrics such as accuracy, sensitivity, specificity, and confusion matrix were used to assess model performance. The model achieved a training accuracy of 100% and a testing accuracy of 91%.

## Building the Web App:
The web app was built using Flask, allowing for easy deployment and interaction with the pneumonia detection model. Flask simplifies web application development and provides flexibility in handling requests and rendering templates.


