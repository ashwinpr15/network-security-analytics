# Network Security Analytics (IDS)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![Security](https://img.shields.io/badge/Domain-Cybersecurity-red?style=flat-square)

A Deep Learning-based **Intrusion Detection System (IDS)** that demonstrates the use of **1D Convolutional Neural Networks (Conv1D)** applied to network connection records.

**Note:** This repository includes a synthetic data generator that mimics the feature structure of the **NSL-KDD** dataset for reproducibility.

## ⚡ Technical Highlights
* **Architecture:** 1D-CNN for temporal/spatial feature extraction from connection records.
* **Preprocessing:** MinMax scaling for continuous variables and One-Hot encoding for categorical fields (`protocol_type`, `service`, `flag`).
* **Evaluation:** Precision, Recall, F1-Score, and ROC-AUC are computed on a stratified test split to ensure robust performance assessment on imbalanced data.

## 📊 Model Architecture
The model treats network connection records as sequential 1D signals, allowing the CNN to learn local correlations between traffic features.

| Layer | Type | filters/Units | Activation |
| :--- | :--- | :--- | :--- |
| **Input** | Feature Vector | - | - |
| **Conv1D** | Spatial Filtering | 64 | ReLU |
| **BatchNormalization**| Stabilization | - | - |
| **MaxPooling** | Downsampling | - | - |
| **Dense** | Classification | 1 (Binary) | Sigmoid |

## 📂 Repository Structure
* `src/model.py`: TensorFlow/Keras CNN architecture definition.
* `src/data_utils.py`: Synthetic data generator and preprocessing pipeline.
* `train.py`: Training script with stratified splitting, early stopping, and metrics.
* `requirements.txt`: Project dependencies.

## 🚀 Usage
1. Install dependencies:
   `pip install -r requirements.txt`
2. Train the IDS model:
   `python train.py`
