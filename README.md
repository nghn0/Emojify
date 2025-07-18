# Emojify: Real-Time Facial Emotion Detection and Emoji Reactions

A deep learning project that enables **real-time facial emotion recognition** and responds with matching **emoji reactions**. Built using **CNNs, attention mechanisms (SE blocks), and Vision Transformers (ViTs)**, the project demonstrates the strengths of modern AI for human-computer interaction through facial expressions.

## 🔍 Abstract

Facial emotion recognition is critical in applications like surveillance, healthcare, driver safety, and entertainment. This project implements and compares three architectures:

1. A baseline CNN
2. An SE-augmented attention CNN
3. A hybrid CNN+Vision Transformer (ViT)

These models were trained and evaluated on the **FER2013** and a subset of **AffectNet** datasets using techniques like **focal loss**, **data augmentation**, and **class weighting**. Real-time inference is achieved using **OpenCV** to overlay detected emotions live from webcam input.

## 📂 Datasets Used

### FER2013
- 35,887 grayscale images (48x48 px)
- 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

> ⚠️ **Important**:  
> Download the FER2013 dataset from [this Kaggle link](https://www.kaggle.com/datasets/msambare/fer2013)  
> Once downloaded, extract and place it inside your working directory like so:
>
> ```
> emojify/
> ┗ data/
>    ┣ train/
>    ┗ test/
> ```

---

### AffectNet (subset)
- 12,815 RGB images
- Same 7 emotions (excluding “contempt”)

> ⚠️ **Important**:  
> Download the AffectNet dataset (subset) from [this Kaggle link](https://www.kaggle.com/datasets/mstjebashazida/affectnet)  
> Once downloaded, extract and place it inside your working directory like below and delete the contempt folder from train and test subfolder:
>
> ```
> emojify/
> ┗ affdata/
>    ┣ train/
>    ┗ test/
> ```

---



All images were resized to 48x48, normalized, and augmented to ensure training efficiency and model generalization.

## 📁 Directory Files

This section outlines the functionality of each Python script in the project and the dataset it is based on.

### 🖥️ GUI Scripts (Real-Time Detection)

- `gui_base_cnn.py`  
  → Real-time facial emotion detection using the **Base CNN** model trained on **FER2013**.

- `gui_attn_cnn.py`  
  → Real-time detection using the **Attention-enhanced CNN (SE blocks)** model trained on **FER2013**.

- `gui_cnn_vit.py`  
  → Real-time detection using the **CNN + Vision Transformer hybrid** model trained on **FER2013**.

---

### 🧪 Training Scripts (FER2013 Dataset)

- `train_base_cnn.py`  
  → Trains a **Base CNN model** on the **FER2013** dataset.

- `train_attn_cnn.py`  
  → Trains a **CNN model with Squeeze-and-Excitation attention** on **FER2013**.

- `train_cnn_vit.py`  
  → Trains a **CNN + Vision Transformer hybrid model** on **FER2013**.

---

### 🧪 Training Scripts (AffectNet Dataset)

- `train2_base_cnn.py`  
  → Trains a **Base CNN model** on the **AffectNet** dataset.

- `train2_cnn_attn.py`  
  → Trains a **CNN model with attention layers** (multi-head attention) on **AffectNet**.

- `train2_cnn_vit.py`  
  → Trains a **CNN + Vision Transformer hybrid model** on **AffectNet**.


## 🧠 Model Architectures

### 1. Baseline CNN
- 3 convolutional layers + max pooling
- Dense layer (1024 units) + Softmax
- ~5M parameters

### 2. Attention CNN (SE-CNN)
- Adds Squeeze-and-Excitation (SE) blocks
- Emphasizes important facial features
- ~6.2M parameters

### 3. CNN + Vision Transformer (ViT)
- CNN extracts local features
- Transformer captures global context
- ~9M parameters

## ⚙️ Training Details

- Optimizer: Adam (`lr=0.0001`, `decay=1e-6`)
- Epochs: 75
- Batch Size: 64
- Class weights: Based on inverse class frequencies
- Loss Function: Focal loss (to handle class imbalance)

## 🎥 Real-Time Deployment

- **Face Detection**: OpenCV Haar Cascade / DNN
- **Inference Pipeline**:
  1. Capture webcam frame
  2. Detect face
  3. Preprocess (resize to 48x48 grayscale)
  4. Predict emotion
  5. Overlay corresponding emoji on the frame

- **Performance**:
  - Baseline CNN and SE-CNN run smoothly in real-time.
  - CNN+ViT performs well but with slight lag.

## 📊 Results Summary

| Dataset   | Model         | Accuracy | Validation Accuracy |
|-----------|---------------|----------|----------------------|
| FER2013   | Base CNN      | 52.66%   | 58.87%               |
| FER2013   | Attention CNN | 51.35%   | 57.97%               |
| FER2013   | CNN+ViT       | 52.41%   | 55.60%               |
| AffectNet | Base CNN      | 44.99%   | 48.29%               |
| AffectNet | Attention CNN | 34.67%   | 39.83%               |
| AffectNet | CNN+ViT       | 40.22%   | 38.85%               |

## 💡 Conclusion

- The **Baseline CNN** provided the best trade-off between accuracy and efficiency for real-time deployment.
- The **SE-CNN** added interpretability by focusing on key facial regions.
- The **CNN+ViT** hybrid showed robustness but was computationally more intensive.

## 🔭 Future Work

- Explore lightweight models like **MobileNet** and **EfficientNet** for edge deployment.
- Implement **multimodal emotion recognition** combining facial expressions with voice or body language.
- Expand dataset diversity to improve cross-population generalization.



---

## 🧑‍💻 Authors

- Nithish Gowda H N (1RVU23CSE317)
- Prajna (1RVU23CSE339)
- Pratham Rajesh Vernekar (1RVU23CSE349)
- Nandan Kumar (1RVU23CSE299)

---

## 📚 References

1. Goodfellow et al., "Challenges in Representation Learning", FER2013  
2. Hu et al., "Squeeze-and-Excitation Networks", 2017  
3. Dosovitskiy et al., "An Image is Worth 16x16 Words", ViT, 2020  
4. Viola & Jones, "Rapid Object Detection", 2001  
5. He et al., "Deep Residual Learning for Image Recognition", 2016

