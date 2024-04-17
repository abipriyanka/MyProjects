# Hand Gesture Recognition using PyTorch

## Introduction
Gesture recognition, particularly static hand gesture recognition, holds significant potential in various fields, including robotics, virtual reality, and sign language interpretation. This project focuses on recognizing hand gestures corresponding to music player commands, such as "next," "pause," and "volume_up."

## Procedure
### Data Collection and Transformation
A dataset comprising hand gesture images is collected and transformed using resizing, grayscale conversion, and normalization techniques. These transformations help improve model accuracy and robustness.

### Model Architecture
The model employs a Convolutional Neural Network (CNN), specifically ResNet-18, for feature extraction and classification. Transfer learning is utilized, where the pre-trained ResNet-18 model is fine-tuned on the hand gesture recognition task.

### Training
The model is trained using stochastic gradient descent (SGD) optimization with backpropagation. Hyperparameters such as learning rate and batch size are tuned to optimize performance. Cross Entropy Loss function is used as the loss criterion during training.

### Validation and Testing
Validation is performed to tune hyperparameters and prevent overfitting, while testing evaluates the model's performance on unseen data.

## Observation
The model achieves high training and validation accuracies across epochs, indicating effective learning. Confusion matrix analysis reveals the model's performance in classifying different hand gestures.

Results observed:

| Epoch | Training Loss | Training Accuracy (%) | Validation Loss | Validation Accuracy (%) |
|-------|---------------|-----------------------|-----------------|-------------------------|
|   1   |    1.0161     |         65.57         |      0.2730     |          97.07          |
|   2   |    0.3000     |         93.08         |      0.3547     |          87.62          |
|   3   |    0.1871     |         95.06         |      0.1222     |          94.79          |
|   4   |    0.1157     |         96.71         |      0.1041     |          97.07          |


## Future Scope
Several avenues for further development are identified, including dataset expansion, real-time detection implementation, and deployment on mobile devices. Improving the user interface and expanding gesture recognition capabilities are also highlighted.

