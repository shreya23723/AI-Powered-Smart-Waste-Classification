from pyspark.sql import SparkSession
from pyspark.ml.image import ImageSchema
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder.appName("GarbageClassifier").getOrCreate()

# Define dataset paths
train_dir = "dataset/DATASET/TRAIN/"
test_dir = "dataset/DATASET/TEST/"

# Load dataset
def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize images to 128x128
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load Organic & Recyclable images
train_images_O, train_labels_O = load_images(os.path.join(train_dir, "O"), 0)
train_images_R, train_labels_R = load_images(os.path.join(train_dir, "R"), 1)

# Merge datasets
train_images = np.concatenate((train_images_O, train_images_R), axis=0)
train_labels = np.concatenate((train_labels_O, train_labels_R), axis=0)

# Normalize images
train_images = train_images / 255.0

# Define CNN Model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  # Binary classification (O = 0, R = 1)
])

# Compile Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# Save Model
model.save("saved_model/garbage_classification.h5")
print("Model training completed and saved.")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()