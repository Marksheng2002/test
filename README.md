# VGG Feature Extraction and Random Forest Ensembles on Tiny ImageNet

This repository contains the implementation for **Question B** of the coursework.  
It uses a subset of **Tiny ImageNet-200**, a pretrained **VGG16** network, and **Random Forest** classifiers to study different ensemble strategies (B.1–B.5).

All experiments are implemented in a single Jupyter Notebook.

---

## 1. Environment

- Python 3.8+
- Recommended: Jupyter Notebook / JupyterLab

Required Python packages:

- `torch`, `torchvision`
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `seaborn` (for confusion matrix heatmaps, optional but used)

Example installation:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm seaborn
```


## 2. Dataset

The experiments use **Tiny ImageNet-200**.

After extraction, the dataset directory should look like:

```text
tiny-imagenet-200/
├── train/
│   ├── n01443537/
│   │   └── images/*.JPEG
│   ├── ...
├── val/
│   ├── images/*.JPEG
│   └── val_annotations.txt
├── test/
│   └── images/*.JPEG
├── wnids.txt
└── words.txt
```

## 3. Overall Workflow (B.1–B.5)

The notebook is organised into five main sections corresponding to the coursework tasks.

---

### B.1 – Data Preparation

**Goals**

- Randomly sample **10 classes** from Tiny ImageNet using a fixed random seed (e.g., `2025`).
- Construct a balanced subset:
  - **1,000 training images** (≈ 100 per class)
  - **200 validation/test images** (≈ 20 per class)
- Apply VGG-compatible preprocessing to all sampled images:
  - Resize to **224 × 224**
  - Normalize with ImageNet mean and std:
    - `mean = [0.485, 0.456, 0.406]`
    - `std  = [0.229, 0.224, 0.225]`
- Visualise **10 random training images** with labels and show **per-class sample counts**.

**Key variables**

- `RANDOM_SEED`: fixed seed for reproducibility.
- `train_processed` / `val_processed`: lists of  
  `(img_tensor, label_idx, path)` for all 1,000/200 images after resize + normalization.

---

### B.2 – VGG Feature Extraction and PCA

**Goals**

- Load **VGG16** with ImageNet pretrained weights and **freeze all parameters**.
- Extract at least three feature views for each image:
  - **Conv3-GAP**: global average pooled feature from a Conv3 block (e.g., 256-dim).
  - **Conv4-GAP**: global average pooled feature from a Conv4 block (e.g., 512-dim).
  - **AvgPool-Flat**: flattened output after `vgg.avgpool` (25088-dim).
- Compute feature matrices for:
  - **1,000 training images**
  - **200 validation/test images**
- Apply **PCA** to reduce dimensionality (e.g., to 128 dimensions) and analyse explained variance.

**Key variables**

- Raw features:
  - `train_conv3`, `train_conv4`, `train_flat`
  - `val_conv3`,   `val_conv4`,   `val_flat`
- PCA-reduced features:
  - `train_conv3_pca`, `train_conv4_pca`, `train_flat_pca`
  - `val_conv3_pca`,   `val_conv4_pca`,   `val_flat_pca`
- Labels:
  - `y_train`, `y_val`
- PCA statistics:
  - original vs reduced dimensionality
  - cumulative explained variance per view

---

### B.3 – Random Forest Ensemble A: Same Data, Different Features

**Goals**

- Train three **Random Forest** classifiers on the **same 1,000 training images**, using different VGG feature views:
  - RF on **Conv3-GAP PCA** features
  - RF on **Conv4-GAP PCA** features
  - RF on **AvgPool-Flat PCA** features
- Evaluate all models on the same **200 test images**.
- Form **Ensemble A** by **averaging class probability outputs** of the three models.
- Report **accuracy** and **confusion matrix**.

**Key variables**

- Models:
  - `rf_conv3`, `rf_conv4`, `rf_flat`
- Predictions:
  - `pred_conv3`, `pred_conv4`, `pred_flat` – single-view RF predictions
  - `ensemble_prob` – averaged probabilities
  - `pred_ens` – Ensemble A final predictions


---

### B.4 – Random Forest Ensemble B: Different Data, Different Features

**Goals**

- Build **Ensemble B** with higher diversity by randomising both **data** and **feature views** at the tree level:
  - For each tree:
    - Draw a **bootstrap sample** from the 1,000 training images.
    - Randomly select one of the three feature views (**Conv3**, **Conv4**, **AvgPool**).
    - Train a shallow RF (or a single-tree RF) on this bootstrap sample and feature view.
- Aggregate predictions over all trees by **probability averaging**.
- Evaluate **Ensemble B** on the same **200 test images**.
- Compare its performance with **Ensemble A** in terms of:
  - accuracy
  - bias/variance behaviour (qualitative)
  - tree correlation (qualitative)

**Key variables**

- `val_probs_all`: stacked probabilities from all trees on the validation set.
- `ensembleB_prob`: mean probability across trees.
- `pred_ensB`: Ensemble B predictions.

---

### B.5 – Comparative Analysis and Ablation

**Goals**

- Compare **Ensemble A** and **Ensemble B** on the 200 test images using:
  - overall **accuracy**
  - **per-class F1 scores** and **Macro-F1**
- Perform an **ablation study** on the number of trees:
  - Train single Random Forest models on one chosen feature view  
    (e.g., **AvgPool-Flat PCA** features).
  - Vary `n_estimators` (e.g., 50 / 100 / 200 / 300).
  - Record **accuracy** and **Macro-F1** for each setting.

**Key variables**

- A vs B metrics:
  - `acc_A`, `acc_B`
  - `macroF1_A`, `macroF1_B`
  - `f1_A`, `f1_B` (per-class F1 arrays)
- Per-class summary table:
  - columns: `Class`, `F1_A`, `F1_B`, `Δ(B−A)`
- Ablation results:
  - `ablation`: DataFrame with columns `#Trees`, `Accuracy`, `Macro-F1`


