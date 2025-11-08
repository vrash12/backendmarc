"""
Flask microservice exposing /health, /features, /predict, /retrain.
- Trains a RandomForest on a CSV (auto-builds from DB if missing).
- Cloud Run friendly (/tmp for writes, listens on $PORT).
- Flexible CORS via env: ALLOW_ALL_CORS=1 or CORS_ORIGINS="https://foo,https://bar".
"""

import os
import logging
from threading import Lock
from typing import List

import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# Your dataset builder (must exist in the repo)
from build_training_dataset import build_dataset

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
app.logger.setLevel(logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")))

# Cloud Run writable dir
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/tmp")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Paths
MODEL_PATH = os.path.join(STORAGE_DIR, "model.pkl")
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(STORAGE_DIR, "training_data.csv"))
PDF_PATH = os.path.join(STORAGE_DIR, "decision_tree.pdf")  # optional, may not exist

# Model size
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "100"))

# Control whether we’re allowed to build the CSV from DB when missing
ALLOW_DB_BUILD = os.environ.get("ALLOW_DB_BUILD", "1") == "1"

# CORS configuration
ALLOW_ALL_CORS = os.environ.get("ALLOW_ALL_CORS", "0") == "1"
if ALLOW_ALL_CORS:
    # Quick testing: allow any browser origin (no cookies needed for this API)
    CORS(app, resources={r"/*": {"origins": "*"}})
else:
    CORS_ORIGINS = [
        o.strip() for o in os.environ.get(
            "CORS_ORIGINS",
            # put your real frontend origin(s) here, comma-separated
            "http://localhost:8000"
        ).split(",") if o.strip()
    ]
    CORS(app, origins=CORS_ORIGINS)

REDIRECT_URL = os.environ.get("REDIRECT_URL", "http://127.0.0.1:8000")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_csv(csv_path: str = DATA_PATH):
    """
    Make sure the training CSV exists. If missing and allowed, build from DB.
    """
    if os.path.isfile(csv_path):
        return
    if not ALLOW_DB_BUILD:
        raise FileNotFoundError(
            f"{csv_path} does not exist and ALLOW_DB_BUILD=0. "
            "Provide a CSV via DATA_PATH or enable ALLOW_DB_BUILD=1."
        )
    app.logger.info("Training data CSV not found, building from database...")
    build_dataset(csv_path)
    app.logger.info("Training data built successfully: %s", csv_path)


def train_and_save() -> RandomForestClassifier:
    """
    Train RandomForest on the CSV and persist the model to MODEL_PATH.
    """
    ensure_csv(DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Training dataset is empty. No responses found in database.")
    if len(df) < 10:
        raise ValueError(f"Insufficient training data: only {len(df)} rows. Need at least 10.")
    if "tech_field_id" not in df.columns:
        raise ValueError("Missing 'tech_field_id' column in training data.")

    X = df.drop(columns=["tech_field_id"])
    y = df["tech_field_id"]

    if X.empty or X.shape[1] == 0:
        raise ValueError("No feature columns found in training data.")

    app.logger.info("Training RandomForest(n_estimators=%d) with %d rows, %d features",
                    N_ESTIMATORS, len(df), X.shape[1])
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
    clf.fit(X, y)
    dump(clf, MODEL_PATH)
    app.logger.info("Model saved to %s", MODEL_PATH)
    return clf


def get_model() -> RandomForestClassifier:
    """
    Load an existing model or train a new one.
    """
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    return train_and_save()


# Lazy model so startup is instant (no work at import)
_clf = None
_clf_lock = Lock()

def _lazy_model() -> RandomForestClassifier:
    global _clf
    if _clf is None:
        with _clf_lock:
            if _clf is None:
                _clf = get_model()
    return _clf


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    """
    If templates/index.html exists, render it; otherwise return JSON.
    """
    templates_dir = os.path.join(app.root_path, "templates")
    index_tpl = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_tpl):
        return render_template("index.html", redirect_url=REDIRECT_URL)
    return jsonify({
        "status": "ok",
        "service": "career-comm-ml",
        "redirect_url": REDIRECT_URL
    })


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.get("/export_tree")
def export_tree():
    """
    Return a previously generated PDF (optional).
    Note: This app trains a RandomForest; it does not generate a PDF by itself.
    """
    if os.path.exists(PDF_PATH):
        return send_file(PDF_PATH, as_attachment=True)
    return ("PDF not found", 404)


@app.get("/features")
def features():
    """
    Expose the required feature order for building request vectors client-side.
    """
    try:
        model = _lazy_model()
        feats: List[str] = model.feature_names_in_.tolist()  # type: ignore[attr-defined]
        return jsonify({"features": feats})
    except Exception as e:
        app.logger.exception("/features failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.post("/predict")
def predict():
    """
    Body: { "features": [ ... numeric values ... ] }
    Returns: { <class_id>: probability, ... }
    """
    payload = request.get_json(silent=True) or {}
    feats = payload.get("features", [])

    try:
        model = _lazy_model()
        expected = len(model.feature_names_in_)  # type: ignore[attr-defined]

        if not isinstance(feats, list):
            return jsonify({"status": "error", "error": "Payload must include a 'features' array."}), 400

        if len(feats) != expected:
            return jsonify({
                "status": "error",
                "error": f"Expected {expected} features in this order: {model.feature_names_in_.tolist()}",  # type: ignore[attr-defined]
            }), 400

        df_feats = pd.DataFrame([feats], columns=model.feature_names_in_)  # type: ignore[attr-defined]
        probs = model.predict_proba(df_feats)[0]
        labels = model.classes_.tolist()

        # JSON-safe: cast keys/values
        response = {int(k): float(v) for k, v in zip(labels, probs.tolist())}
        return jsonify(response)

    except Exception as e:
        app.logger.exception("/predict failed: %s", e)
        return jsonify({"status": "error", "error": str(e)}), 500


@app.post("/retrain")
def retrain():
    """
    Rebuild (if needed) and retrain the model, then load it into memory.
    """
    try:
        global _clf
        _clf = train_and_save()
        return jsonify({
            "status": "retrained",
            "classes": [int(c) for c in _clf.classes_.tolist()],
            "message": "Model retrained successfully"
        })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "suggestion": "Ensure you have completed questionnaires in the database before retraining."
        }), 400
    except Exception as e:
        app.logger.exception("/retrain failed: %s", e)
        return jsonify({"status": "error", "error": f"Unexpected error during retraining: {e}"}), 500


# -----------------------------------------------------------------------------
# Local dev entrypoint (Cloud Run uses gunicorn app:app)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
