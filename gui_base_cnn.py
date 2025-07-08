import tkinter as tk
from tkinter import Label, Button
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import urllib.request
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Auto-download face detection model
def download_model():
    if not os.path.exists('deploy.prototxt.txt'):
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
            'deploy.prototxt.txt'
        )
    if not os.path.exists('res10_300x300_ssd_iter_140000.caffemodel'):
        urllib.request.urlretrieve(
            'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
            'res10_300x300_ssd_iter_140000.caffemodel'
        )

download_model()

# Load DNN face detector
face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Build CNN model
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
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
        Dense(7, activation='softmax')
    ])
    return model

# Load trained model
emotion_model = build_model()
emotion_model.load_weights('emotion_model.weights.h5')
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emoji_dist = {0: "emojis/angry.png", 1: "emojis/disgusted.png", 2: "emojis/fearful.png",
              3: "emojis/happy.png", 4: "emojis/neutral.png", 5: "emojis/sad.png", 6: "emojis/surprised.png"}

cap1 = cv2.VideoCapture(0)
if not cap1.isOpened():
    print(" Error: Cannot open the camera!")
    exit()
show_text = [0]

def show_vid():
    global cap1, show_text
    ret, frame = cap1.read()
    if not ret:
        root.after(10, show_vid)
        return

    frame = cv2.resize(frame, (600, 500))
    (h, w) = frame.shape[:2]

    # DNN face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    max_confidence = 0
    best_box = None

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence and confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype("int")
            max_confidence = confidence

    if best_box is not None:
        (x1, y1, x2, y2) = best_box
        cv2.rectangle(frame, (x1, y1 - 50), (x2, y2 + 10), (255, 0, 0), 2)
        roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        try:
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)
            show_text[0] = int(np.argmax(prediction))
        except:
            pass

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    root.after(20, show_vid)

def show_vid2():
    image_path = emoji_dist.get(show_text[0])
    if image_path and os.path.exists(image_path):
        emoji_img = cv2.imread(image_path)
        emoji_rgb = cv2.cvtColor(emoji_img, cv2.COLOR_BGR2RGB)
        imgtk2 = ImageTk.PhotoImage(image=Image.fromarray(emoji_rgb))
        lmain2.imgtk = imgtk2
        lmain2.configure(image=imgtk2)
        lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))

    root.after(100, show_vid2)

# GUI Setup
root = tk.Tk()
root.title("Photo To Emoji")
root.geometry("1400x900+100+10")
root['bg'] = 'black'

heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 50, 'bold'), bg='black', fg='white')
heading2.pack()

lmain = tk.Label(root, padx=50, bd=10, bg='black')
lmain2 = tk.Label(root, bd=10, bg='black')
lmain3 = tk.Label(root, bd=10, fg="white", bg='black', font=('arial', 45, 'bold'))

lmain.place(x=50, y=250)
lmain3.place(x=960, y=250)
lmain2.place(x=900, y=350)

exitbutton = Button(root, text='Quit', fg="red", command=root.quit, font=('arial', 25, 'bold'))
exitbutton.pack(side="bottom", pady=20)

threading.Thread(target=show_vid, daemon=True).start()
threading.Thread(target=show_vid2, daemon=True).start()

root.mainloop()
cap1.release()
