#Import libraries
import tensorflow as tf
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths
DATASET_PATH = "dataset/"
MODEL_PATH = "saved_model/animal_classifier.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = datagen.flow_from_directory(DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", subset="training")

# Build model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_data, epochs=10)

# Save model
os.makedirs("saved_model", exist_ok=True)
model.save(MODEL_PATH)

# Load trained model for testing
model = tf.keras.models.load_model(MODEL_PATH)

# Predict on a new image
def predict_animal(image_path):
    class_labels = os.listdir(DATASET_PATH)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_idx = np.argmax(predictions)

    return class_labels[class_idx]

# Test on a sample image
test_image = "test_images/sample.jpg"  # Provide test image path
predicted_class = predict_animal(test_image)
print(f"{predicted_class}")