"""
Flask microservice exposing /predict and /retrain,
auto-building the CSV if missing. Cloud Run friendly.
"""

import os
from threading import Lock

import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

from build_training_dataset import build_dataset

# Load env from .env when present (handy for local dev)
load_dotenv()

# -----------------------------------------------------------------------------
# Flask app + Cloud Run basics
# -----------------------------------------------------------------------------
app = Flask(__name__)

# Cloud Run has a read-only FS; only /tmp is writable at runtime.
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/tmp")
os.makedirs(STORAGE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(STORAGE_DIR, "model.pkl")
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(STORAGE_DIR, "training_data.csv"))
PDF_PATH = os.path.join(STORAGE_DIR, "decision_tree.pdf")  # optional, may not exist

# Allow overriding RF size via env (default 100)
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "100"))

# CORS origins (comma-separated list)
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "https://career-comm-main-laravel.onrender.com,https://ccsuggest.netlify.app,http://localhost:8000",
    ).split(",")
    if o.strip()
]
CORS(app, origins=CORS_ORIGINS)

REDIRECT_URL = os.environ.get("REDIRECT_URL", "http://127.0.0.1:8000")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_csv(csv_path: str = DATA_PATH):
    """
    Make sure the CSV exists. If missing, build it from the database via
    build_training_dataset.build_dataset().
    """
    if not os.path.isfile(csv_path):
        app.logger.info("Training data CSV not found, building from database...")
        build_dataset(csv_path)
        app.logger.info(f"Training data built successfully: {csv_path}")


def train_and_save() -> RandomForestClassifier:
    """
    Train a RandomForest on the training CSV and save it to MODEL_PATH.
    Returns the fitted classifier.
    """
    try:
        ensure_csv(DATA_PATH)
        df = pd.read_csv(DATA_PATH)

        if df.empty:
            raise ValueError("Training dataset is empty. No responses found in database.")
        if len(df) < 10:
            raise ValueError(f"Insufficient training data: only {len(df)} records found. Need at least 10.")
        if "tech_field_id" not in df.columns:
            raise ValueError("Missing 'tech_field_id' column in training data.")

        X = df.drop("tech_field_id", axis=1)
        y = df["tech_field_id"]

        if X.empty or X.shape[1] == 0:
            raise ValueError("No feature columns found in training data.")

        app.logger.info(f"Training RandomForest(n_estimators={N_ESTIMATORS}) "
                        f"with {len(df)} rows and {X.shape[1]} features.")
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
        clf.fit(X, y)
        dump(clf, MODEL_PATH)
        app.logger.info(f"Model saved to {MODEL_PATH}")
        return clf
    except Exception as e:
        app.logger.exception(f"Training failed: {e}")
        raise


def get_model() -> RandomForestClassifier:
    """
    Load a saved model if present; otherwise train and save a new one.
    """
    if os.path.exists(MODEL_PATH):
        return load(MODEL_PATH)
    return train_and_save()


# Lazy model so container can start immediately (no blocking work at import).
_clf = None
_clf_lock = Lock()


def _lazy_model() -> RandomForestClassifier:
    global _clf
    if _clf is None:
        with _clf_lock:
            if _clf is None:  # double-checked locking
                _clf = get_model()
    return _clf


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def index():
    """
    Simple root. If templates/index.html exists, render it;
    otherwise return a JSON OK so the root never 500s.
    """
    templates_dir = os.path.join(app.root_path, "templates")
    index_tpl = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_tpl):
        return render_template("index.html", redirect_url=REDIRECT_URL)
    return jsonify({"status": "ok", "redirect_url": REDIRECT_URL})


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.get("/export_tree")
def export_tree():
    """
    Optional: returns a previously generated decision_tree.pdf if present.
    (This app trains a RandomForest; PDF export is not generated here.)
    """
    if os.path.exists(PDF_PATH):
        return send_file(PDF_PATH, as_attachment=True)
    return ("PDF not found", 404)


@app.get("/features")
def features():
    """
    Expose feature column names expected by the model (helps your frontend).
    """
    try:
        model = _lazy_model()
        return jsonify({"features": model.feature_names_in_.tolist()})
    except Exception as e:
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
        expected = len(model.feature_names_in_)

        if len(feats) != expected:
            return jsonify({
                "status": "error",
                "error": f"Expected {expected} features in this order: {model.feature_names_in_.tolist()}",
            }), 400

        df_feats = pd.DataFrame([feats], columns=model.feature_names_in_)
        probs = model.predict_proba(df_feats)[0]
        labels = model.classes_.tolist()
        # Ensure JSON-serializable keys
        response = {int(k): float(v) for k, v in zip(labels, probs.tolist())}
        return jsonify(response)
    except Exception as e:
        app.logger.exception(f"/predict failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.post("/retrain")
def retrain():
    """
    Rebuild (if needed) and retrain the model, then reload it in-memory.
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
        app.logger.exception(f"/retrain failed: {e}")
        return jsonify({"status": "error", "error": f"Unexpected error during retraining: {e}"}), 500


# -----------------------------------------------------------------------------
# Local dev entrypoint (Cloud Run uses gunicorn app:app)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
