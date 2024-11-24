import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt

DATASET_PATH = "C:\\Users\\Sumit\\OneDrive\\Desktop\\soil\\archive\\Soil types"
LABELS = ["Black Soil", "Cinder Soil", "Laterite Soil", "Yellow Soil", "Peat Soil"]
IMG_SIZE = 128  # Resize images to 128x128

# Data augmentation function
def augment_image(img):
    augmented_images = [img]
    augmented_images.append(cv2.flip(img, 1))  # Horizontal flip
    augmented_images.append(cv2.GaussianBlur(img, (5, 5), 0))  # Gaussian blur
    return augmented_images


# Function to load images and preprocess them
def load_images_from_folders(base_path, labels, img_size):
    X, y = [], []
    for label_idx, label in enumerate(labels):
        folder_path = os.path.join(base_path, label)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                img = cv2.imread(file_path)
                img = cv2.resize(img, (img_size, img_size))
                for aug_img in augment_image(img):  # Apply augmentation
                    features = aug_img.flatten()
                    X.append(features)
                    y.append(label_idx)
            except Exception as e:
                print(f"Error loading image {file_name}: {e}")
    return np.array(X), np.array(y)


# Load dataset
X, y = load_images_from_folders(DATASET_PATH, LABELS, IMG_SIZE)

# Normalize features (scale pixel values between 0 and 1)
X = X / 255.0


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
rf_model = grid_search.best_estimator_

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Function to predict soil type from an image
def predict_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    features = img_resized.flatten().reshape(1, -1) / 255.0

    with open("soil_recognition_model.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    labels = data["labels"]
    prediction = model.predict(features)[0]
    return labels[prediction]

