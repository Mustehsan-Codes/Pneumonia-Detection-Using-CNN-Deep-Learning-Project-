# Pneumonia Detection from Chest X-Ray using CNN 🩻🧠

This project implements a deep learning model using Convolutional Neural Networks (CNN) to detect pneumonia from chest X-ray images. It utilizes the **TensorFlow/Keras** framework and the **Chest X-Ray Images (Pneumonia)** dataset. The model is trained to classify images into two classes: **NORMAL** and **PNEUMONIA**, achieving ~90% accuracy on the test set.

---

## 📁 Dataset

- **Name:** Chest X-Ray Images (Pneumonia)
- **Primary Source:** [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Alternative Source (Mendeley):** [Mendeley Data – Chest X-ray Images (Pneumonia)](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- **Structure:**
  - `train/`: 5,216 images
  - `val/`: 16 images
  - `test/`: 624 images
- **Classes:** `NORMAL`, `PNEUMONIA`
- **Format:** JPG Images (RGB)

---

## 🧪 Technologies & Libraries

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Google Colab

---

## 🛠️ Features and Workflow

### ✅ **1. Data Preprocessing**
- Resized images to **150x150 pixels**
- Normalized pixel values (scaled to range 0–1)
- Applied **image augmentation** (rotation, zoom, horizontal flip)
- Used `flow_from_directory` for efficient data loading and label assignment

### ✅ **2. CNN Architecture**
- 3 × `Conv2D` + `ReLU` + `MaxPooling`
- `Flatten` layer
- `Dense` (ReLU) + `Dropout`
- Output `Dense` layer with `Sigmoid` activation for binary classification

### ✅ **3. Model Compilation**
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Learning Rate:** 0.001
- **Batch Size:** 32
- **Epochs:** 10

### ✅ **4. Model Evaluation**
- Training Accuracy: ~98%
- Validation Accuracy: ~90%
- Test Accuracy: ~90%
- Plotted loss and accuracy graphs to confirm convergence
- Confusion Matrix and Classification Report showed high precision and recall

---

## 📊 Results

- The model performs well on unseen data, showing good generalization.
- No major overfitting due to the use of dropout and image augmentation.

---

## 📌 Folder Structure

