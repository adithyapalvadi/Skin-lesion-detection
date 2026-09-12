# Skin Lesion Detection & Classification

This repository contains deep learning pipelines for multi-class skin lesion classification on the **HAM10000** dataset (ISIC 2018 Task 3).

---

## 🔬 Supported Architectures

The project evaluates and compares multiple convolutional and transformer-based architectures:

- **ResNet-50**
- **DenseNet-121**
- **EfficientNet-B0**
- **Swin Transformer Tiny (Swin-T)**
- **Vision Transformer (ViT-B/16)**
- **Feature Fusion Models** (Channel Attention, Spatial Attention, Cross Attention, and MLP Fusion)

---

## 📊 Dataset & Diagnoses (7 Classes)

1. **akiec** — Actinic Keratoses and Intraepithelial Carcinoma
2. **bcc** — Basal Cell Carcinoma
3. **bkl** — Benign Keratosis-like Lesions
4. **df** — Dermatofibroma
5. **mel** — Melanoma
6. **nv** — Melanocytic Nevi
7. **vasc** — Vascular Lesions

---

## 📈 Evaluation Metrics

- Multi-Class Accuracy & Balanced Accuracy
- Macro Precision, Recall, and F1-Score
- Per-Class Specificity & Confusion Matrix
- Multi-Class AUROC & ROC Curves
- 95% Bootstrap Confidence Intervals

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install torch torchvision pandas numpy scikit-learn matplotlib tqdm pillow
```

### 2. Dataset Setup
- Download the **HAM10000** dataset and place `HAM10000_metadata.csv` in the root folder.
- Place the image files in `ISIC_ES/ISIC2018_Task3_Training_Input/`.

### 3. Running the Notebook
Open and run [EMBEDDED__PROJECT.ipynb](EMBEDDED__PROJECT.ipynb) in VS Code or Jupyter Lab.
