# Early-Lung-Cancer-Diagnosis-System
- Published at **ICAIN 2025**: “Comparative Study of CNN Variants Vs ViT for Lung Cancer Diagnosis”.   - Fine-tuned **deep learning models** (VGG16, ResNet, DenseNet, ViT) for medical imaging.   - Evaluated with **accuracy, precision, recall, F1-score, AUC**.   - Proposed an **AI-driven diagnostic pipeline** for healthcare.  

---

# 📘 Project 3: **Early Lung Cancer Diagnosis (CNN vs ViT)**  
📂 `Lung-Cancer-Diagnosis/README.md`  

```markdown
# 🫁 Early Lung Cancer Diagnosis – CNN vs Vision Transformer

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Medical Imaging](https://img.shields.io/badge/Domain-Medical%20AI-green)

## 📖 Overview
This project explores **deep learning architectures** (CNNs and Vision Transformers) for **early lung cancer detection** using histopathological image datasets.  
The research was published at **ICAIN 2025** and is currently under review in an **SCIE Springer Journal**.

## 🛠️ Tech Stack
- **Frameworks:** TensorFlow, PyTorch, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Dataset:** LC25000 (Histopathological Images)  

## 🚀 Features
- Compared **CNN variants** (VGG16, ResNet50, DenseNet, InceptionNet, EfficientNet) vs **Vision Transformer (ViT)**  
- Evaluated using **Accuracy, Precision, Recall, F1-score, AUC**  
- Proposed an **AI-driven diagnostic pipeline** for healthcare  

## 📊 Workflow
```mermaid
flowchart LR
A[Input Lung Image] --> B[Preprocessing]
B --> C[Model Training - CNN/ViT]
C --> D[Evaluation Metrics]
D --> E[Diagnosis Results]

📈 Results

Accuracy: CNN ~92%, ViT ~95%

Improved precision & recall in early detection

Published at ICAIN 2025 (BITS Pilani, Dubai)

📂 Repository Structure
Lung-Cancer-Diagnosis/
 ├── data/          # Dataset links & preprocessing
 ├── notebooks/     # Training notebooks
 ├── models/        # Saved models
 ├── results/       # Confusion matrix, ROC curves
 └── README.md
