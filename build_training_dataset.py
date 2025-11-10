"""
Exports questionnaire responses + final tech-field label to a CSV,
auto-detecting the answer column and encoding non-numerics.

Defaults to Hostinger "marcky" DB and is Cloud Run friendly (/tmp path).
All values can be overridden with environment variables.
"""

import os
import pandas as pd
import mysql.connector

# ---- Hostinger defaults (override via env as needed) -------------------------
DB_DEFAULTS = {
    "DB_HOST": "217.21.80.1",          # or "217.21.80.1"
    "DB_PORT": 3306,
    "DB_NAME": "u782952718_marcky",
    "DB_USER": "u782952718_marcky",        # change if your DB user differs
    "DB_PASS": "Allenturing123.",                          # set in Cloud Run/Secret Manager
    "DB_SSL_DISABLED": "true",              # Hostinger commonly OK with this
}

def _env(name: str) -> str:
    default = DB_DEFAULTS[name]
    return os.environ.get(name, str(default))

# -----------------------------------------------------------------------------

def build_dataset(csv_path: str = "/tmp/training_data.csv"):
    # Read connection settings (env overrides defaults above)
    db_host = _env("DB_HOST")
    db_port = int(_env("DB_PORT"))
    db_name = _env("DB_NAME")
    db_user = _env("DB_USER")
    db_pass = _env("DB_PASS")
    ssl_disabled = _env("DB_SSL_DISABLED").lower() in ("1", "true", "yes")

    # 1) Connect to MySQL (Hostinger)
    db = mysql.connector.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_pass,
        database=db_name,
        ssl_disabled=ssl_disabled,
    )

    try:
        cursor = db.cursor()

        # 2) Introspect `responses` for the answer column
        cursor.execute(f"SHOW COLUMNS FROM `{db_name}`.`responses`")
        cols = [row[0] for row in cursor.fetchall()]
        for candidate in ("answer", "response", "value", "response_value", "answer_text"):
            if candidate in cols:
                answer_col = candidate
                break
        else:
            raise RuntimeError(
                f"Could not find a response column in `{db_name}.responses`; found columns: {cols}"
            )

        # 3) Query joined data (schema-qualified for safety)
        query = f"""
            SELECT r.user_id,
                   r.question_id,
                   r.`{answer_col}` AS answer,
                   rec.tech_field_id
            FROM `{db_name}`.`responses` r
            JOIN `{db_name}`.`recommendations` rec ON rec.user_id = r.user_id
        """
        df = pd.read_sql(query, con=db)

    finally:
        cursor.close()
        db.close()

    if df.empty:
        raise ValueError("No rows returned from responses/recommendations; cannot build training set.")

    # 4) Preserve raw before numeric casting
    df["answer_raw"] = df["answer"]
    numeric = pd.to_numeric(df["answer_raw"], errors="coerce")
    non_numeric_mask = numeric.isna() & df["answer_raw"].notna()
    df["answer"] = numeric

    # Encode non-numeric answers per question deterministically
    if non_numeric_mask.any():
        encoded_chunks = []
        for qid, g in df.loc[non_numeric_mask, ["question_id", "answer_raw"]].groupby("question_id"):
            codes, _ = pd.factorize(g["answer_raw"].astype(str), sort=True)
            encoded_chunks.append(pd.Series(codes, index=g.index))
        if encoded_chunks:
            enc_series = pd.concat(encoded_chunks).astype(float)
            df.loc[enc_series.index, "answer"] = enc_series

    # 5) Pivot into feature matrix
    features = df.pivot_table(
        index="user_id",
        columns="question_id",
        values="answer",
        aggfunc="first",
    )
    features.columns = [f"Q{int(c)}" for c in features.columns]
    features = features.sort_index(axis=1).fillna(-1)  # numeric matrix

    # 6) Label per user
    labels = df.groupby("user_id")["tech_field_id"].first()

    # 7) Combine + write CSV
    dataset = features.join(labels).reset_index(drop=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    dataset.to_csv(csv_path, index=False)
    return dataset
