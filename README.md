# 🚀 Real-Time Data Drift Detector (PSI-Based)

## 🧠 Overview
This project is a production-ready **Data Drift Detection System** that monitors incoming data and identifies distribution shifts using the **Population Stability Index (PSI)**.

It is deployed as a **Flask web application on Render**, allowing users to upload new datasets and instantly detect drift compared to the training baseline.

---

## 🎯 Problem Statement
Machine Learning models degrade over time when real-world data changes. This issue, known as **data drift**, can silently reduce model accuracy without any visible errors.

This project solves that by:
- Continuously comparing new data with training data
- Detecting distribution changes
- Alerting users with clear drift metrics

---

## ⚙️ Features
- 📊 Feature-level drift detection using PSI
- 📁 Upload CSV for real-time analysis
- 🚨 Drift classification:
  - **No Drift** (< 0.1)
  - **Moderate Drift** (0.1 – 0.25)
  - **High Drift** (> 0.25)
- 🌐 Flask-based web application
- ☁️ Deployed on Render
- 🧩 Handles:
  - Missing values
  - Column mismatch
  - Invalid data

---

## 🏗️ Tech Stack
- **Python**
- **pandas, numpy**
- **scikit-learn**
- **Flask (Backend API)**
- **Gunicorn (Production server)**
- **Render (Deployment)**

---

## 🔍 How It Works

1. **Training Phase**
   - Train ML model on dataset
   - Save baseline dataset (`train.csv`)

2. **Incoming Data**
   - User uploads new dataset via UI

3. **Drift Detection**
   - System compares feature distributions
   - Uses PSI to quantify change

4. **Output**
   - Displays PSI per feature
   - Classifies drift level

---

## 📈 PSI Interpretation

| PSI Value | Drift Level        |
|----------|-------------------|
| < 0.1    | No Drift          |
| 0.1–0.25 | Moderate Drift    |
| > 0.25   | High Drift 🚨     |

---

## 📁 Project Structure
