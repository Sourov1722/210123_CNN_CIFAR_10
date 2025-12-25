# Vehicles & Animals Image Classification with CIFAR-10

This project implements a complete Convolutional Neural Network (CNN) image classification pipeline using PyTorch, trained on the CIFAR-10 dataset and evaluated on both the standard test set and real-world smartphone images. The model is trained on the CIFAR-10 dataset and evaluated on real-world smartphone images to analyze generalization performance.

---

## Dataset

### Standard Dataset
CIFAR-10 (10 Classes)

### Classes Used
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

### Custom Dataset
Real-world smartphone images captured by the author (some collected from the internet) and stored in: data/custom/


---

## Project Structure
├── data/ <br>
│ └── custom/<br>
├── model/<br>
│ └── 210123.pth<br>
├── CNN_Image_Classification.ipynb<br>
├── README.md<br>



---

## Model Architecture

A Convolutional Neural Network (CNN) implemented using torch.nn.Module.

### Key Layers
- Convolution + ReLU
- Max Pooling
- Fully Connected Layers

---

## Training

- Dataset automatically downloaded using torchvision.datasets
- Images preprocessed using torchvision.transforms
- Model trained on CIFAR-10 training set

---

## Training Results

The model was trained for 10 epochs on the CIFAR-10 training set.

### Final Epoch Log

Epoch [10/10]<br>
Loss: 0.2490<br>
Accuracy: 0.9132<br>


### Training Loss vs Epochs
![Training Loss](images/train_loss.png)

### Training Accuracy vs Epochs
![Training Accuracy](images/train_accuracy.png)

---

## Evaluation & Results

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Key Observations
- Strong performance on structured object classes (automobile, truck, ship)
- Some confusion among visually similar animal classes (cat, dog, deer, bird)

---

## Visual Error Analysis
Three randomly misclassified samples from the CIFAR-10 test set:

![Error Analysis](images/error_analysis.png)

---

## Real-World Smartphone Image Predictions

Predictions on custom smartphone images stored in `data/custom/`:

![Custom Predictions](images/custom_predictions.png)

### Observations
- Vehicles are classified with high confidence
- Animal classes occasionally show confusion
- Confidence varies due to domain shift

---

## Key Takeaways

- CNN successfully learns CIFAR-10 visual patterns
- Training is stable and well-converged
- Real-world testing reveals generalization limits
- End-to-end deep learning workflow demonstrated

---

##  How to Run (Google Colab)

1. Open [210123_CNN_CIFAR_10.ipynb](https://colab.research.google.com/drive/1EC1eMeMdnBCPLxWkJZ-uXZggeKQxKJEn?usp=sharing) in Google Colab

2. Select Runtime → Run all


# Author
# Sohrab Hossain Sourov
# Student ID: 210123



