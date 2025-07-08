import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     LayerNormalization, MultiHeadAttention, Add, Reshape)
from tensorflow.keras import backend as K
from PIL import Image
import os

# Emotion setup
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surpriced']
emoji_dict = {e: f'emojis/{e}.png' for e in emotion_labels}

# Load DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# Build CNN + ViT model
def transformer_block(x, num_heads, ff_dim):
    embed_dim = x.shape[-1]
    attn_input = LayerNormalization(epsilon=1e-6)(x)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(attn_input, attn_input)
    x = Add()([x, attn_output])
    ffn_input = LayerNormalization(epsilon=1e-6)(x)
    ffn_output = tf.keras.Sequential([
        Dense(ff_dim, activation='relu'),
        Dense(embed_dim),
    ])(ffn_input)
    return Add()([x, ffn_output])

def build_model():
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

    x = Conv2D(128, kernel_size=1, padding='same')(x)
    x = Reshape((6 * 6, 128))(x)
    x = transformer_block(x, num_heads=4, ff_dim=64)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(7, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=output)

model = build_model()
model.load_weights("cnn_vit_emotion_model.weights.h5")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))  # Mean subtraction
    face_net.setInput(blob)
    detections = face_net.forward()

    emotion = "neutral"

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            x, y = max(0, x), max(0, y)
            x1, y1 = min(w, x1), min(h, y1)

            face = frame[y:y1, x:x1]
            if face.size == 0:
                continue

            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48)).astype("float32") / 255.0
            gray = np.expand_dims(gray, axis=[0, -1])

            preds = model.predict(gray, verbose=0)[0]
            emotion = emotion_labels[np.argmax(preds)]
            print("[INFO] Emotion:", emotion)

            # Draw rectangle + text
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Draw emoji
            emoji_path = emoji_dict.get(emotion)
            if emoji_path and os.path.exists(emoji_path):
                emoji_img = Image.open(emoji_path).resize((200, 200))
                emoji_np = cv2.cvtColor(np.array(emoji_img), cv2.COLOR_RGBA2BGRA)
                eh, ew, _ = emoji_np.shape
                frame[10:10 + eh, 10:10 + ew] = emoji_np[:, :, :3]

    cv2.imshow("Emotion Detection (DNN Face)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
