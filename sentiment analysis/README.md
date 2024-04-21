# Toxic Comment Classification

This project aims to classify toxic comments into six different categories: toxic behavior, severe toxicity, obscene language, threat, insult, and identity hate. The model is trained on the Jigsaw Toxic Comment Classification dataset.

## Dataset
The dataset used in this project is the Jigsaw Toxic Comment Classification dataset, available on [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). The dataset contains comments from Wikipedia's talk page edits, and each comment is labeled for toxicity.

## Requirements
This project requires the following Python libraries:
- Python 3
- TensorFlow 2.x
- Pandas
- Gradio


## Steps Involved

### 1. Data Preparation
- Load the dataset using Pandas.
- Extract comment text (`X`) and corresponding labels (`y`).

### 2. Text Preprocessing
- Use TensorFlow's `TextVectorization` layer to convert raw text into numerical format.
- Tokenize and vectorize the text data.
- Prepare the data pipeline using `tf.data.Dataset`.

### 3. Model Architecture
- Build a deep learning model using TensorFlow's Keras API.
- The model architecture includes:
  - An embedding layer.
  - A bidirectional LSTM layer.
  - Dense layers for feature extraction.
  - A final dense layer with sigmoid activation for multi-label classification.

### 4. Model Training
- Compile the model with appropriate loss and optimizer.
- Train the model on the prepared data.
- Evaluate the model's performance using precision, recall, and accuracy metrics.

### 5. Model Deployment
- Save the trained model for future use.
- Create a Gradio interface for the trained model to make predictions on user input.


