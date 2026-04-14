<div align="center">
  <h1>📝 Domain Scorer Pro</h1>
  <p><b>Automated Essay Scoring (AES) System</b></p>
  
  ![Python](https://img.shields.io/badge/python-v3.8%2B-blue.svg?style=for-the-badge)
  ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
  ![LightGBM](https://img.shields.io/badge/LightGBM-F37021?style=for-the-badge)
  ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
</div>

<br>

> A high-performance, AI-driven essay scoring system built to grade student essays with human-level reliability, running entirely on consumer CPU hardware.

---

## ✨ Overview

Manual essay grading is highly subjective and time-consuming. **Domain Scorer Pro** resolves this challenge by utilizing advanced NLP embeddings (`all-mpnet-base-v2`) and a highly optimized `LightGBM Regressor` ensemble to predict human grader scores. The system achieves an overall **Quadratic Weighted Kappa (QWK) of 0.9260**, surpassing the professional-grade industry target of 0.90.

For a full deep dive into the methodology, data pipeline, normalization strategies, and comprehensive performance metrics, please read the [Detailed Project Report](./PROJECT_REPORT.md).

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| 🎯 **High Precision Scoring** | Achieves a QWK of ≥ 0.90 on cross-validation across 8 uniquely tuned essay sets. |
| ⚡ **CPU-Optimized Engine** | A fully operational 5-Fold Stratified K-Fold ensemble pipeline trains in under 30 minutes without GPU dependency. |
| 🧠 **Hybrid Extractor** | Combines 768-dimensional semantic embeddings (sentence-transformers) with 9 hand-crafted linguistic metrics. |
| 💻 **Real-Time Interface** | Features a premium, glassmorphism-styled UI integrated directly via a responsive FastAPI backend. |

---

## 🛠️ System Architecture

1. **Semantic Encoding**: Raw text is directly passed through the frozen `all-mpnet-base-v2` transformer model to create a rich semantic layout.
2. **Linguistic Engineering**: Hand-crafted metrics calculate lexical diversity, text density, and structural syntactic complexity.
3. **Machine Learning Backbone**: A 5-Fold Gradient Boosted ensemble framework (`LightGBM`) processes the hybrid 777-dimensional array to predict normalized scores across all prompts.
4. **Denormalization Engine**: Accurately maps boundary predictions back into their context-specific score criteria limits limit scales (ranging from 1-6 up to 10-60).

---

## ⚙️ Get Started Locally

Follow these steps to deploy the inference API and premium frontend locally:

### 1. Clone the repository
```bash
git clone https://github.com/Prince-410/Automated-Essay-Scoring.git
cd Automated-Essay-Scoring
```

### 2. Install required dependencies
Make sure you have python installed. Running this application requires a few essential data science tools. 
```bash
pip install pandas numpy scikit-learn lightgbm sentence-transformers fastapi uvicorn joblib
```

### 3. Launch the AI Web Server
Start the ASGI server for the FastAPI real-time REST API framework:
```bash
python server.py
```
*Your frontend dashboard and application will instantly be accessible dynamically at `http://127.0.0.1:8000`.*

---
<div align="center">
  <i>Developed to bring unbiased, objective scoring to educational technology.</i>
</div>
