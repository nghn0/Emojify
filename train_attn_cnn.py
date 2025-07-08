import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape, Multiply, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import tensorflow as tf
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

# Data Augmentation & Normalization (More augmentation for minority classes)
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

# Define Squeeze-and-Excitation Block (Attention mechanism)
def squeeze_excite_block(input_tensor, ratio=16):
    filters = K.int_shape(input_tensor)[-1]  # Number of channels
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(input_tensor)  # Squeeze: global average pooling
    se = Reshape(se_shape)(se)  # Reshape to (1, 1, filters)
    se = Dense(filters // ratio, activation='relu')(se)  # Fully connected layer
    se = Dense(filters, activation='sigmoid')(se)  # Fully connected layer to get the attention weights
    se = Multiply()([input_tensor, se])  # Recalibrate the input feature map
    return se

# Define the model with functional API
inputs = Input(shape=(48, 48, 1))

# Conv Layer 1
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs) # Extracts features
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x) # drops samples to reduce overfitting

# Attention Block 1
x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = squeeze_excite_block(x)  # Helps in including important features and removing least imp features

# Next Conv Layers
x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

# Flatten and Dense Layers
x = Flatten()(x) # Converts 2d to 1d
x = Dense(1024, activation='relu')(x) # To connect all the features
x = Dropout(0.5)(x)
x = Dense(7, activation='softmax')(x)

# Define the model
emotion_model = Model(inputs=inputs, outputs=x)

cv2.ocl.setUseOpenCL(False) # to disable tensorflow+GPU

emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful",
    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
}

# Define Focal Loss (Alternative to categorical cross-entropy) cce - helps in class detection
def focal_loss(alpha=0.25, gamma=2.0): # helps in class detection when dataset is imbalanced
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.mean(weight * ce, axis=-1)
    return loss

# Compile model
emotion_model.compile(
    loss=focal_loss(alpha=0.25, gamma=2.0),  # Use Focal Loss for class imbalance
    optimizer=Adam(learning_rate=0.0001, decay=1e-6), # optimizer for  declaring learning rate and reducing learning rate
    metrics=['accuracy']
)

# Train model with class weights
emotion_model.fit(
    train_generator,
    epochs=75,  # Increased epochs to learn from augmented data
    validation_data=validation_generator,
    class_weight=class_weights  # Apply class weights
)

# Save model weights
emotion_model.save_weights('emotion_model_attention.weights.h5')

