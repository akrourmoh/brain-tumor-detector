# Brain Tumor Detection

An end-to-end deep learning application that detects brain tumors from MRI images using a Convolutional Neural Network (CNN) built with TensorFlow/Keras, deployed as a web application with Flask.

---

## Live Demo

**[brain-tumor-detector-production.up.railway.app](https://brain-tumor-detector-production.up.railway.app)**

Upload an MRI image and get an instant prediction — tumor or no tumor, with confidence score.

---

## Project Overview

This project automatically classifies brain MRI images as tumorous or non-tumorous using a CNN trained from scratch.

The full pipeline includes:
- Data exploration and class imbalance analysis
- Data augmentation (253 → 2065 images)
- Image preprocessing (ROI extraction, resize, normalization)
- CNN training with Dropout regularization
- Model evaluation (accuracy, F1-score, confusion matrix)
- Web application deployment (Flask + HTML/CSS/JavaScript)

---

## Results

| Metric | Score |
|---|---|
| Accuracy | ~88.7% |
| F1-score | 0.88 |
| Precision | ~0.89 |
| Recall | ~0.87 |

---

## Project Structure

```
brain-tumor-detection/
│
├── app.py                        ← Flask backend
├── requirements.txt              ← Python dependencies
│
├── models/
│   └── best_model.h5             ← Trained CNN model
│
├── templates/
│   └── index.html                ← Frontend HTML
│
├── static/
│   ├── style.css                 ← CSS styling
│   └── script.js                 ← JavaScript logic
│
├── 01_data_exploration.ipynb     ← Dataset analysis
├── 02_data_augmentation.ipynb    ← Data augmentation pipeline
├── 03_preprocessing.ipynb        ← ROI extraction + preprocessing
├── 04_model_training.ipynb       ← CNN architecture + training
└── 05_evaluation.ipynb           ← Model evaluation + metrics
```

---

## Dataset

- **Source:** Kaggle — Brain MRI Images for Brain Tumor Detection
- **Original size:** 253 images (155 tumor / 98 no tumor)
- **After augmentation:** 2065 images
- **Classes:** Binary — `yes` (tumor) / `no` (no tumor)

---

## Model Architecture

```
Input (240×240×3)
    ↓
ZeroPadding2D
    ↓
Conv2D (32 filters, 7×7) → BatchNormalization → ReLU
    ↓
MaxPooling2D (4×4) → Dropout (0.25)
    ↓
MaxPooling2D (4×4)
    ↓
Flatten → Dense (32, ReLU) → Dropout (0.5)
    ↓
Dense (1, Sigmoid)  →  0 = No Tumor / 1 = Tumor
```

**Why a simple CNN?**
Transfer learning models (ResNet50, VGG16) overfitted on this small dataset. A lightweight custom CNN with Dropout regularization performs better.

---

## Preprocessing Pipeline

Every MRI image goes through 3 steps before prediction:

1. **ROI Extraction** — detects and crops only the brain region, removing useless black background
2. **Resize** — standardizes all images to 240×240×3
3. **Normalization** — scales pixel values from [0-255] to [0.0-1.0]

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/akrourmoh/brain-tumor-detector.git
cd brain-tumor-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python app.py
```

**4. Open your browser**
```
http://localhost:5000
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Image Processing | OpenCV, NumPy |
| Backend | Flask, Werkzeug |
| Frontend | HTML5, CSS3, JavaScript |
| Deployment | Docker, Railway |

---

## Author

**Mohammed Akrour**
- Master's in Biometrics & Intelligent Vision — UPEC Paris-Est Créteil
- [LinkedIn](https://linkedin.com/in/mohammed-akrour)
- [GitHub](https://github.com/akrourmoh)
