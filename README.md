# 🏎️ Animals & Vehicles Classification with CIFAR-10 🐈
**Student ID:** 210123  
**Course:** Neural Networks & Deep Learning  

This project implements a custom Convolutional Neural Network (CNN) using **PyTorch** to classify images from the CIFAR-10 dataset. The project evaluates the model's performance on both standard test data and real-world images captured via a smartphone.

---

## 📊 Dataset Overview
- **Standard Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (60,000 images, 10 classes)
- **Categories:** `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- **Custom Dataset:** 10 real-world images captured by smartphone, stored in the `dataset/` folder for testing generalization.

---

## 🏗️ Model Architecture (CifarNet_210123)
The model is designed with a unique block-based structure to ensure high accuracy while preventing overfitting:

- **Feature Extractor:**
  - 3 Convolutional Blocks with **Batch Normalization** for faster convergence.
  - ReLU Activation and Max-Pooling (2x2) layers.
  - Output feature map size: 128 channels of 4x4.
- **Classifier:**
  - Fully Connected (Dense) layers.
  - **Dropout (0.3)** to improve generalization on real-world images.
  - Final output layer with 10 neurons (Softmax).



[Image of a convolutional neural network architecture for image classification]


---

## 📈 Project Workflow & Results

### 1. Training Performance
The model was trained for 10 epochs using the Adam optimizer. 
- **Learning Rate:** 0.001
- **Loss Function:** Cross-Entropy Loss
- **Final Accuracy:** Achieved high validation accuracy with stable loss reduction.

### 2. Evaluation
- **Confusion Matrix:** Analyzes per-class performance, highlighting strengths in vehicle detection and minor confusion between similar animal classes (e.g., cat vs dog).
- **Error Analysis:** Visualizes the top 3 misclassified images from the test set to understand model limitations.

### 3. Real-World Testing
The model processes images from the `dataset/` folder, applying standard normalization:
- Outputs class prediction.
- Displays confidence score (%).
- Shows the model's ability to handle "Domain Shift" (from low-res CIFAR to high-res phone photos).

---

## 📂 Repository Structure
```text
├── dataset/             # Real-world smartphone images for testing
├── model/               # Contains the trained weights (210123.pth)
├── 210123.ipynb         # Main Google Colab Notebook
└── README.md            # Project documentation
