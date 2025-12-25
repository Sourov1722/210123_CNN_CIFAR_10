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
├── dataset/ <br>
├── model/<br>
│ └── 210123.pth<br>
├── 210123_CNN_CIFAR_10.ipynb<br>
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


<img width="981" height="374" alt="download" src="https://github.com/user-attachments/assets/f649ab0c-18ae-4536-a08f-d54dacde25cb" />


---

## Evaluation & Results

### Confusion Matrix
<img width="853" height="766" alt="download" src="https://github.com/user-attachments/assets/583df261-3876-4c66-9dbd-6ef5413973e4" />



### Key Observations
- Strong performance on structured object classes (automobile, truck, ship)
- Some confusion among visually similar animal classes (cat, dog, deer, bird)

---

## Visual Error Analysis
Three randomly misclassified samples from the CIFAR-10 test set:

<img width="950" height="336" alt="download" src="https://github.com/user-attachments/assets/fd614fc2-4b8d-4242-af0a-972e4268ce07" />


---

## Real-World Smartphone Image Predictions

Predictions on custom smartphone images stored in `data/custom/`:


<img width="1182" height="581" alt="download" src="https://github.com/user-attachments/assets/75a45823-4409-493d-934f-902f53419d78" />


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



