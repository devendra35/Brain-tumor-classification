# 🧠 Brain Tumor Classification & Localization

An AI-powered **brain MRI analysis application** that classifies MRI scans into four categories and localizes tumor regions using deep learning.

The application combines **EfficientNetB3** for image classification with a **Lightweight U-Net** for tumor localization and provides an interactive interface built with **Streamlit**.

> ⚠️ **Research/Educational Use Only:** This project is not a medical diagnostic system and should not replace evaluation by a qualified medical professional.

## 🚀 Live Demo

**Try the application:**
https://brain-tumor-classification-35.streamlit.app/

## 📌 Overview

Brain MRI analysis is an important application of computer vision and deep learning. This project demonstrates an end-to-end workflow for analyzing brain MRI images using trained deep learning models.

The system performs two main tasks:

### 1. Brain Tumor Classification

An **EfficientNetB3** model analyzes the uploaded MRI image and predicts one of four classes:

* 🟣 Glioma
* 🔵 Meningioma
* ⚪ No Tumor
* 🟢 Pituitary

### 2. Tumor Localization

When a tumor class is predicted, a **Lightweight U-Net** model is used to generate a tumor segmentation mask.

The application displays:

* Original MRI
* Predicted tumor mask
* Tumor localization overlay
* Mask probability
* Tumor pixel count
* Estimated mask area

The classification model uses `300 × 300` input images, while the segmentation model uses `128 × 128` inputs.

## ✨ Features

* 🧠 Deep learning-based MRI classification
* 🔬 Four-class tumor classification
* 🎯 Tumor region localization
* 🖼️ MRI image upload
* 📊 Class probability visualization
* 🎚️ Adjustable segmentation threshold
* 📈 Prediction confidence
* 🔍 Original, mask, and overlay visualization
* ⚡ Interactive Streamlit interface

## 🏗️ System Architecture

```text
                 Brain MRI Image
                       │
                       ▼
              Image Preprocessing
                       │
                       ▼
              ┌─────────────────┐
              │  EfficientNetB3 │
              │   Classifier    │
              └────────┬────────┘
                       │
                Predicted Class
                       │
              ┌────────┴────────┐
              │                 │
          No Tumor          Tumor Class
              │                 │
              ▼                 ▼
          Final Result    Lightweight U-Net
                                │
                                ▼
                         Tumor Segmentation
                                │
                                ▼
                       Localization Overlay
```

The application skips the segmentation stage when the classifier predicts `No Tumor`.

## 🤖 Models

### EfficientNetB3

Used for four-class brain MRI classification.

**Input:** `300 × 300`

**Classes:**

```text
Glioma
Meningioma
No Tumor
Pituitary
```

### Lightweight U-Net

Used for tumor segmentation and localization.

**Input:** `128 × 128`

The trained model files are stored in the `models/` directory.

## 📊 Model Performance

| Task           | Metric     |     Result |
| -------------- | ---------- | ---------: |
| Classification | Accuracy   | **88.33%** |
| Segmentation   | Dice Score | **70.72%** |
| Segmentation   | IoU        | **59.91%** |

These are the performance values currently displayed by the deployed application.

## 🔍 Application Workflow

1. Upload a brain MRI image in JPG, JPEG, or PNG format.
2. The image is converted to RGB.
3. The MRI is resized for classification.
4. EfficientNetB3 predicts the tumor class.
5. The application displays the predicted class and confidence.
6. If a tumor is detected, the image is processed by the Lightweight U-Net.
7. U-Net generates a tumor probability map.
8. A threshold is applied to create the segmentation mask.
9. The mask is resized to the original image dimensions.
10. The application generates a tumor localization overlay.

## 📂 Project Structure

```text
Brain-tumor-classification/
│
├── models/
│   ├── best_efficientnetb3_final.keras
│   └── best_light_unet_128_final.keras
│
├── notebooks/
│   └── 01_brain_tumor_classification.ipynb
│
├── app.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

The repository contains the trained EfficientNetB3 and Lightweight U-Net models, the main Jupyter notebook, and the Streamlit application.

## 🛠️ Tech Stack

| Category            | Technology         |
| ------------------- | ------------------ |
| Programming         | Python             |
| Deep Learning       | TensorFlow / Keras |
| Classification      | EfficientNetB3     |
| Segmentation        | Lightweight U-Net  |
| Image Processing    | Pillow             |
| Numerical Computing | NumPy              |
| Visualization       | Matplotlib         |
| Web Application     | Streamlit          |
| Development         | Jupyter Notebook   |

The repository's current dependencies include Streamlit, TensorFlow, NumPy, Pillow, and Matplotlib.

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/devendra35/Brain-tumor-classification.git
cd Brain-tumor-classification
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🖼️ Using the App

### Upload an MRI

The application accepts:

```text
JPG
JPEG
PNG
```

Upload an MRI image using the **Upload Brain MRI** section.

### Analyze

Click:

```text
🔍 Analyze MRI
```

The application then performs classification and, when appropriate, tumor segmentation.

### View Results

For classification, the application displays:

* Predicted class
* Confidence
* Class probabilities

For tumor predictions, it additionally displays:

* Tumor mask
* Tumor localization overlay
* Mask maximum probability
* Tumor pixel count
* Mask area
* Raw U-Net probability map

## 🎚️ Segmentation Threshold

The application provides an adjustable tumor-mask threshold from **0.10 to 0.90**, with a default value of **0.50**.

A higher threshold produces a stricter mask, while a lower threshold produces a more sensitive mask.

## 📓 Notebook

The model development and experimentation notebook is available in:

```text
notebooks/01_brain_tumor_classification.ipynb
```

## 🎯 Project Goals

The main goals of this project are to:

* Explore deep learning for medical image analysis
* Classify brain MRI images into tumor categories
* Detect and localize tumor regions
* Build an accessible AI-assisted interface
* Demonstrate an end-to-end computer vision workflow

## 🔮 Future Improvements

* Improve classification performance
* Improve tumor segmentation accuracy
* Add Grad-CAM / explainable AI
* Add model confidence analysis
* Add more robust image preprocessing
* Evaluate on additional datasets
* Add more comprehensive validation
* Improve visualization and reporting

## ⚠️ Disclaimer

This application is intended strictly for **research and educational purposes**.

The predictions generated by the models are **not medical diagnoses**. They should not be used as a substitute for professional medical advice, diagnosis, or treatment.

Always consult a qualified medical professional for medical interpretation of MRI scans.

## 👨‍💻 Author

### Devendra Khanal

**AI / Machine Learning / Deep Learning Developer**

GitHub:
https://github.com/devendra35

## ⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub.

---

**Built with Python, TensorFlow, Keras, and Streamlit.**
