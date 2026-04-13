import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# -------------------------------
# Load Training Data (Baseline)
# -------------------------------
TRAIN_PATH = "artifacts/train.csv"

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError("artifacts/train.csv not found")

train_df = pd.read_csv(TRAIN_PATH)

# Select only numeric columns and remove target
numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
if "target" in numerical_cols:
    numerical_cols.remove("target")


# -------------------------------
# PSI Function (YOUR LOGIC)
# -------------------------------
def calculate_psi(baseline_series, new_series, num_bins=10):
    all_data = pd.concat([baseline_series, new_series])
    min_val = all_data.min()
    max_val = all_data.max()

    bins = np.linspace(min_val, max_val, num_bins + 1)

    baseline_counts = np.histogram(baseline_series, bins=bins)[0]
    new_counts = np.histogram(new_series, bins=bins)[0]

    epsilon = 1e-6

    baseline_prop = (baseline_counts / len(baseline_series)) + epsilon
    new_prop = (new_counts / len(new_series)) + epsilon

    psi = np.sum((new_prop - baseline_prop) * np.log(new_prop / baseline_prop))

    return psi


# -------------------------------
# Drift Status Function
# -------------------------------
def get_drift_status(psi_value):
    if psi_value < 0.1:
        return "No Drift"
    elif psi_value < 0.25:
        return "Moderate Drift"
    else:
        return "High Drift"


# -------------------------------
# Core Drift Detection
# -------------------------------
def detect_drift(new_df):
    results = []

    for col in numerical_cols:
        if col not in new_df.columns:
            continue

        baseline = train_df[col].dropna()
        current = new_df[col].dropna()

        if len(current) < 2:
            results.append({
                "feature": col,
                "psi": None,
                "status": "Insufficient Data"
            })
            continue

        psi_value = calculate_psi(baseline, current)
        status = get_drift_status(psi_value)

        results.append({
            "feature": col,
            "psi": round(psi_value, 4),
            "status": status
        })

    return results


# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file name"})

    try:
        new_df = pd.read_csv(file)

        # Keep only numeric columns
        new_df = new_df.select_dtypes(include=np.number)

        if new_df.empty:
            return jsonify({"error": "No numeric columns found"})

        results = detect_drift(new_df)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------
# Run (Render Compatible)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
