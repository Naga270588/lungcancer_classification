# lungcancer_classification
# Deep Learning Models for Image Classification

This repository contains implementations of various deep learning architectures using TensorFlow/Keras for image classification tasks. Each model is trained and evaluated with 5-Fold Cross-Validation (5FCV) and/or train-test-validation splitting. Some models are also enhanced with attention mechanisms and hybrid architectures.

---

## Repository Structure

Each folder represents a specific deep learning model or a hybrid combination. Inside each folder, you will find one or more `.ipynb` notebooks used for training, testing validation and evaluation.

### Standard Architectures
- `AlexNet`
- `DenseNet121`
- `DenseNet169`
- `EfficientNetB0`
- `EfficientNetB3`
- `InceptionResnetV2`
- `InceptionV3`
- `MobileNet`
- `MobileNetV2`
- `ResNet50`
- `ResNet101`
- `Vgg16`
- `vgg19`
- `Xception`

###  Hybrid and Enhanced Architectures
- `InceptionResNet + GRU`: Combines CNN and RNN for temporal modeling. (Also have an Notebook implemention of XAI)
- `ResNet + MultiHeadAttention layer`: ResNet backbone enhanced with multi-head attention for better feature extraction.
- `VGG + MobileNet`: A fusion of VGG and MobileNet features.
- `VGG + ResNet`: Combines VGG and ResNet architectures.
- `MobileNet + ResNet`: Combines MobileNet and ResNet architectures.
- `Xception + DenseNet`: Combines Xception and DenseNet architectures.
###  Dataset Splitting & Evaluation
- `'Dataset split'`: Notebook(s) for organizing train/test/validation splits.
- `'Train-test-val(accuracy,loss).ipynb'`: A generic notebook to evaluate models with accuracy and loss metrics.

---

##  Cross-Validation Strategy

Many models follow **5-Fold Cross-Validation (5FCV)** for robust performance evaluation. This helps mitigate overfitting and improves generalization.

---

##  Evaluation Metrics

Each notebook provides:
- Training and validation accuracy/loss plots
- Confusion matrices
- Model summary and parameter count
- Final test accuracy

---

## ⚙️ Requirements

All notebooks are implemented in **Python 3.10** using **TensorFlow** and **Keras**. To get started:

```bash
pip install tensorflow matplotlib numpy scikit-learn
