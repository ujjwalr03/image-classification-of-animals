# Import Libraries
import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import glob

DATASET_PATH = "dataset/"
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff', '*.webp']

# Data Preprocessing (REPLACE this section)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)


# Use enhanced augmentation only for training data
train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# Upload an Image for Testing
image_path = next((img for ext in image_extensions for img in glob.glob(ext)), None)

# Load Model for Prediction
model = tf.keras.models.load_model("animal_classifier.h5")

# Predict on Uploaded Image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image at {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Load the best model instead of the last one
model = tf.keras.models.load_model("best_animal_model.h5")

# Get class mapping from the training generator
class_indices = train_data.class_indices
class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping

# Make prediction
processed_img = preprocess_image(image_path)
if processed_img is not None:
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_idx]

    print(f"{predicted_class}")