import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten, Conv2D,
                                     MaxPooling2D, GlobalAveragePooling1D, Reshape, LayerNormalization, MultiHeadAttention, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# Paths
train_dir = 'affdata/train'
val_dir = 'affdata/test'

# Detect class distribution
classes = [cls for cls in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cls))]
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in classes}
total_samples = sum(class_counts.values())
class_weights = {i: total_samples / (len(class_counts) * count) for i, (_, count) in enumerate(class_counts.items())}
num_classes = len(classes)

# Data generators (RGB input now)
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
    target_size=(48, 48),
    batch_size=64,
    color_mode="rgb",  # RGB for ViT
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="rgb",  # RGB for ViT
    class_mode='categorical'
)

# Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.mean(weight * ce, axis=-1)
    return loss

# Vision Transformer patch encoder
def transformer_block(x, num_heads, projection_dim):
    # Layer Norm
    norm1 = LayerNormalization(epsilon=1e-6)(x)
    # MHA
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(norm1, norm1)
    attention_output = Dropout(0.1)(attention_output)
    # Skip connection
    out1 = Add()([x, attention_output])

    # Feedforward
    norm2 = LayerNormalization(epsilon=1e-6)(out1)
    ff = Dense(projection_dim * 2, activation='relu')(norm2)
    ff = Dense(projection_dim)(ff)
    ff = Dropout(0.1)(ff)

    # Final output
    return Add()([out1, ff])

# Build CNN + ViT model
def build_cnn_vit_model():
    inputs = Input(shape=(48, 48, 3))  # RGB input
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Reshape CNN features to sequence
    shape_before = x.shape
    num_patches = shape_before[1] * shape_before[2]
    patch_dim = shape_before[3]
    x = Reshape((num_patches, patch_dim))(x)

    # Transformer block
    x = transformer_block(x, num_heads=4, projection_dim=patch_dim)
    x = transformer_block(x, num_heads=4, projection_dim=patch_dim)

    # Pooling and final classification
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

# Compile model
model = build_cnn_vit_model()
model.compile(
    loss=focal_loss(alpha=0.25, gamma=2.0),
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Train
model.fit(
    train_generator,
    epochs=75,
    validation_data=validation_generator,
    class_weight=class_weights
)

# Save weights
model.save_weights('cnn_vit_emotion_model_rgb_48x48.weights.h5')
