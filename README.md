# 🐱🐶 Cat and Dog Image Classifier (TensorFlow)

## 📌 Overview
This project builds a **Convolutional Neural Network (CNN)** to classify images
of **cats and dogs** using **TensorFlow 2.0 and Keras**.  
The model is trained on labeled image data and is required to achieve at least
**63% classification accuracy** on unseen test images
(extra credit for reaching **70%+ accuracy**).

The project is designed to be completed using **Google Colaboratory**.

---

## 🎯 Objective
- Classify images as **Cat** or **Dog**
- Use **TensorFlow 2.0** and **Keras**
- Achieve **≥ 63% accuracy** on the test set
- Complete the missing code cells as instructed
- Pass all automated tests included in the notebook

---

## 🧠 Approach Used
- **Image Preprocessing**
  - Image rescaling (0–255 → 0–1)
  - Directory-based data loading using `ImageDataGenerator`

- **Data Augmentation**
  - Random rotations
  - Zooming
  - Flipping
  - Shifting
  - Shearing

- **Model Architecture**
  - Convolutional layers (Conv2D)
  - Pooling layers (MaxPooling2D)
  - Fully connected (Dense) layers
  - ReLU activation
  - Sigmoid output for binary classification

- **Training & Evaluation**
  - Binary cross-entropy loss
  - Adam optimizer
  - Accuracy metric
  - Visualization of training and validation performance

---

## 🛠️ Technologies
- Python 3
- TensorFlow 2.x
- Keras
- Google Colaboratory
- Matplotlib
- NumPy

---
cat-dog-image-classifier/
│
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   ├── test/
│   │   ├── cats/
│   │   └── dogs/
│
├── model/
│   └── cnn_model.py
│
├── utils/
│   └── data_loader.py
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md

---

## ▶️ How to Run
1. Open the provided notebook in **Google Colab**
2. Create a copy in your own account
3. Run each cell in order
4. Complete the required code sections
5. Train the model and evaluate accuracy
6. Enable link sharing before submission

---

## 📊 Model Output
- Training & validation accuracy graphs
- Training & validation loss graphs
- Predictions on 50 unseen test images
- Confidence score for each prediction

---

## 🧪 Testing
- Automated tests are included in the notebook
- Final cell verifies whether the project passes the challenge
- Accuracy must meet or exceed the required threshold

---

## 🏆 Key Learnings
- Building CNNs with TensorFlow and Keras
- Image preprocessing and augmentation
- Preventing overfitting with data augmentation
- Evaluating deep learning models
- Working with real-world image datasets

---

## 📜 License
This project is for **educational purposes only** as part of a machine learning certification.
