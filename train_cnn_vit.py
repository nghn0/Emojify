import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     LayerNormalization, MultiHeadAttention, Add, Reshape, TimeDistributed)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import os

# Define paths
train_dir = 'data/train'
val_dir = 'data/test'

# Class distribution (Based on dataset)
classes = [cls for cls in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, cls))]
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in classes}

# Compute class weights
total_samples = sum(class_counts.values())
class_weights = {i: total_samples / (7 * count) for i, count in enumerate(class_counts.values())}

# Data augmentation
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
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Define Transformer block
def transformer_block(x, num_heads, ff_dim):
    embed_dim = x.shape[-1]
    attn_input = LayerNormalization(epsilon=1e-6)(x) # it normalizes the training by helps to reduce overfitting
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(attn_input, attn_input) # helps in relating each patches
    x = Add()([x, attn_output]) # helps to maintain gradient flow

    ffn_input = LayerNormalization(epsilon=1e-6)(x)
    ffn_output = tf.keras.Sequential([
        Dense(ff_dim, activation='relu'), # helps in finding pattern between  all the connected features
        Dense(embed_dim),
    ])(ffn_input)

    return Add()([x, ffn_output])

# Define CNN + ViT model
input_layer = Input(shape=(48, 48, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)


x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

# ViT Preparation
x = Conv2D(128, kernel_size=1, padding='same')(x)  # Embed dimension = 128
x = Reshape((6 * 6, 128))(x)  # (batch, num_patches=36, embed_dim=128)

# Transformer block
x = transformer_block(x, num_heads=4, ff_dim=64) # helps in detecting the global features

# Flatten and output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(7, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

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
model.compile(
    loss=focal_loss(alpha=0.25, gamma=2.0),
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    metrics=['accuracy']
)

# Train model
model.fit(
    train_generator,
    epochs=75,
    validation_data=validation_generator,
    class_weight=class_weights
)

# Save model weights
model.save_weights('cnn_vit_emotion_model.weights.h5')
