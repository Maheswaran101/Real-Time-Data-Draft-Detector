import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load training dataset
TRAIN_PATH = "train.csv"

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError("train.csv not found. Place it in root directory.")

train_df = pd.read_csv(TRAIN_PATH)

# Keep only numeric columns
train_df = train_df.select_dtypes(include=[np.number]).dropna()


# -------------------------------
# PSI Calculation Function
# -------------------------------
def calculate_psi(expected, actual, bins=10):
    epsilon = 1e-6

    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.linspace(0, 100, bins + 1)
    breakpoints = np.percentile(expected, breakpoints)

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    psi_values = (actual_percents - expected_percents) * np.log(
        (actual_percents + epsilon) / (expected_percents + epsilon)
    )

    return np.sum(psi_values)


# -------------------------------
# Drift Detection Logic
# -------------------------------
def detect_drift(train_df, new_df):
    results = []

    for col in train_df.columns:
        if col not in new_df.columns:
            continue

        psi = calculate_psi(train_df[col], new_df[col])

        if psi < 0.1:
            status = "No Drift"
        elif psi < 0.25:
            status = "Moderate Drift"
        else:
            status = "High Drift"

        results.append({
            "feature": col,
            "psi": round(psi, 4),
            "status": status
        })

    return results


# -------------------------------
# Routes
# -------------------------------

@app.route("/", methods=["GET"])
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

        # Keep numeric columns only
        new_df = new_df.select_dtypes(include=[np.number]).dropna()

        if new_df.empty:
            return jsonify({"error": "Uploaded file has no valid numeric data"})

        results = detect_drift(train_df, new_df)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------
# Run App (Render Compatible)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
