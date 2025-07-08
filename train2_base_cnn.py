import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow as tf

# Define paths
train_dir = 'affdata/train'
val_dir = 'affdata/test'

# Automatically detect class counts from training data
classes = [cls for cls in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cls))]
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in classes}

# Compute class weights
total_samples = sum(class_counts.values())
class_weights = {i: total_samples / (len(class_counts) * count) for i, (_, count) in enumerate(class_counts.items())}

# Data Augmentation & Normalization (grayscale conversion)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),        # Resize 1024x1024 → 48x48
    batch_size=64,
    color_mode="grayscale",      # Convert to grayscale
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),        # Resize 1024x1024 → 48x48
    batch_size=64,
    color_mode="grayscale",      # Convert to grayscale
    class_mode='categorical'
)

# Update number of classes dynamically
num_classes = len(train_generator.class_indices)

# Define Model (grayscale input)
emotion_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),  # Grayscale input (1 channel)
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Output size = number of folders (classes)
])

# Define Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.mean(weight * ce, axis=-1)
    return loss

# Compile model
emotion_model.compile(
    loss=focal_loss(alpha=0.25, gamma=2.0),
    optimizer=Adam(learning_rate=0.0001, weight_decay=1e-6),
    metrics=['accuracy']
)

# Train model
emotion_model.fit(
    train_generator,
    epochs=75,
    validation_data=validation_generator,
    class_weight=class_weights
)

# Save model weights
emotion_model.save_weights('emotion_model_grayscale_48x48.weights.h5')
