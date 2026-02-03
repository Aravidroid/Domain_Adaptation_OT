# 📦 Domain Adaptation using Optimal Transport (MNIST → USPS)

This repository implements a **classical domain adaptation pipeline using Optimal Transport (OT)** to align feature distributions between source and target domains. The project focuses on **Sinkhorn-regularized OT** and evaluates its impact using **distance-based classifiers** under severe domain shift.

The approach is validated on the **MNIST → USPS** benchmark dataset.

---

## 📌 Problem Motivation

Machine learning models often fail when deployed on data distributions different from their training data — a phenomenon known as **domain shift**.

In this project:
- **Source domain:** MNIST (handwritten digits)
- **Target domain:** USPS (handwritten digits)

Despite sharing the same labels (0–9), these datasets differ significantly in:
- Image resolution
- Stroke thickness
- Noise patterns

This leads to **very poor baseline performance**, motivating the need for **domain adaptation**.

---

## 🧠 Methodology Overview

We compare three settings:

### 1️⃣ No Adaptation (Baseline)
- Train on source domain
- Test directly on target domain

### 2️⃣ Global Optimal Transport
- Align overall source and target distributions using Sinkhorn OT
- Ignores class structure

### 3️⃣ Class-wise Optimal Transport (Oracle)
- Perform Sinkhorn OT **within each class**
- Aligns source digit *c* to target digit *c*
- Represents an **upper bound** on adaptation performance (uses target labels)

---

## 🔧 Algorithms Used

### 🔹 Optimal Transport
- Sinkhorn-regularized Optimal Transport
- Cosine distance cost matrix
- Implemented using the `POT (Python Optimal Transport)` library

### 🔹 Classifiers
Distance-based classifiers are used due to their robustness under domain shift:
- **Nearest Class Mean (NCM)**
- **k-Nearest Neighbors (kNN, k=1 and k=5)**

(Logistic Regression was tested but underperformed due to severe misalignment.)

---

## 📊 Experimental Results

| Setting | NCM | kNN-1 | kNN-5 |
|------|------|------|------|
| No Adaptation | 6.56% | 6.89% | 9.06% |
| Global OT | 7.11% | 6.28% | 6.61% |
| Class-wise OT (reg=0.01) | 22.61% | 24.11% | **25.11%** |
| Class-wise OT (reg=0.1) | 22.61% | **24.44%** | 22.56% |
| Class-wise OT (reg=1.0) | 22.61% | 22.56% | 22.61% |

---

## 📁 Project Structure

```
domain_adaptation_OT/
│
├── data/
│   ├── MNIST_vs_USPS.mat
│   ├── split.py
│   ├── source.csv
│   └── target.csv
│
├── dataset.py
├── train.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 📥 Dataset Source

The MNIST → USPS dataset used in this project was obtained from the **Transfer Learning Library** curated by **Jindong Wang**.

- Dataset source:  
  https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md

- Original datasets:
  - **MNIST**: Handwritten digit dataset
  - **USPS**: Handwritten digit dataset (U.S. Postal Service)

The dataset is distributed in `.mat` format and converted to CSV for this project.

---

## 🔄 Data Preprocessing

- Samples in the original `.mat` file are stored as `(features × samples)`
- Data is transposed to `(samples × features)` for scikit-learn compatibility
- Source and target domains are saved as `source.csv` and `target.csv`

Conversion script:
```bash
python data/split.py
```

---

## ▶️ How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Convert dataset (only once)
```bash
python data/split.py
```

### Run experiments
```bash
python main.py
```

---

## 📦 Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- scipy
- pot (Python Optimal Transport)

---

## ⚠️ Limitations

- Class-wise OT uses **true target labels** and represents an oracle upper bound
- Experiments are performed on raw pixel features without feature learning

---

## 🚀 Future Work

- t-SNE visualization before/after OT
- PCA or CNN-based feature extraction
- Extension to unsupervised pseudo-label OT
- Application to security datasets (phishing / malware)

---

## 📚 Citation

If you use this dataset or code in academic work, please cite:

> Jindong Wang et al., *Transfer Learning Library*  
> https://github.com/jindongwang/transferlearning

---

## ⭐ Final Note
This project demonstrates that **class-conditional Optimal Transport** can significantly reduce domain shift under challenging conditions and serves as a clean, interpretable reference implementation for classical domain adaptation.
