import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


# Set the paths to your pneumonia and normal image directories
pneumonia_dir = r"C:\Users\ABI PRIYANKA\Downloads\pnuemonia\train\PNEUMONIA"
normal_dir = r"C:\Users\ABI PRIYANKA\Downloads\pnuemonia\train\NORMAL"

# Set the image dimensions
img_width, img_height = 150, 150

# Function to load and preprocess images
def load_images(path, label):
    images = []
    labels = []
    for image_file in os.listdir(path):
        if image_file.endswith('.jpg') or image_file.endswith('.jpeg'):
            image = cv2.imread(os.path.join(path, image_file))
            image = cv2.resize(image, (img_width, img_height))
            images.append(image)
            labels.append(label)
    return images, labels

# Load pneumonia images
pneumonia_images, pneumonia_labels = load_images(pneumonia_dir, 1)
# Downsample the pneumonia class
pneumonia_indices = np.random.choice(len(pneumonia_images), size=int(len(pneumonia_images) / 2), replace=False)
pneumonia_images = [pneumonia_images[i] for i in pneumonia_indices]
pneumonia_labels = [pneumonia_labels[i] for i in pneumonia_indices]

# Load normal images
normal_images, normal_labels = load_images(normal_dir, 0)

# Upsample the normal class
normal_indices = np.random.choice(len(normal_images), size=len(pneumonia_images), replace=True)
normal_images = [normal_images[i] for i in normal_indices]
normal_labels = [normal_labels[i] for i in normal_indices]

# Combine images and labels
images = pneumonia_images + normal_images
labels = pneumonia_labels + normal_labels


# Convert images and labels to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Flatten the image data
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Preprocess the image data (optional: scale the pixel values to [0, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(x_train, y_train)
rf_predictions = rf_classifier.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:")
print(rf_confusion_matrix)



                 

# Calculate the square root of the number of samples
sqrt_n = int(np.sqrt(x_train.shape[0]))

# Create and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=sqrt_n)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train, y_train)
knn_predictions = knn_classifier.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_confusion_matrix = confusion_matrix(y_test, knn_predictions)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)
print("K-Nearest Neighbors Confusion Matrix:")
print(knn_confusion_matrix)


# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(x_train, y_train)
dt_predictions = dt_classifier.predict(x_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Confusion Matrix:")
print(dt_confusion_matrix)

# Plotting the Confusion Matrix
def plot_confusion_matrix(cm, labels):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Plotting the Random Forest Confusion Matrix
plot_confusion_matrix(rf_confusion_matrix, ['Normal', 'Pneumonia'])

# Plotting the K-Nearest Neighbors Confusion Matrix
plot_confusion_matrix(knn_confusion_matrix, ['Normal', 'Pneumonia'])

# Plotting the Decision Tree Confusion Matrix
plot_confusion_matrix(dt_confusion_matrix, ['Normal', 'Pneumonia'])


# Bar plot of accuracy scores
models = ['Random Forest', 'K-Nearest Neighbors', 'Decision Tree']
accuracy_scores = [rf_accuracy, knn_accuracy, dt_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracy_scores, color='skyblue')
plt.xlabel('Classification Models')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores of Classification Models')
plt.ylim([0, 1])
plt.show()