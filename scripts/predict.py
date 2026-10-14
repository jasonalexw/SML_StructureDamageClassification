"""Prediksi data test menggunakan ensemble CatBoost dan LightGBM."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import (
    CLASSES,
    ID_COLUMN,
    TARGET,
    location_keys,
    make_lgb_matrix,
    raw_lgb_features_transform,
    transform_target_encoding,
    validate_feature_frame,
)
from .preprocess import grade_numbers
from .train import catboost_probabilities, ordinal_probabilities


ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT_DIR / "data" / "test.csv"
SAMPLE_PATH = ROOT_DIR / "data" / "sample_submission.csv"
MODEL_PATH = ROOT_DIR / "models" / "pipeline_model_ensemble.pkl"
SUBMISSION_PATH = ROOT_DIR / "submissions" / "submission.csv"


def predict_ensemble(artifact, X):
    """Menggabungkan prediksi CatBoost dan LightGBM ordinal."""

    cat_probs = catboost_probabilities(
        artifact["catboost_pipeline"],
        X,
    )

    raw_features = raw_lgb_features_transform(
        X,
        artifact["category_maps"],
        artifact["prefix_lengths"],
    )
    encoded_features = transform_target_encoding(
        location_keys(X, artifact["prefix_lengths"]),
        artifact["target_maps"],
    )
    lgb_matrix = make_lgb_matrix(raw_features, encoded_features)
    lgb_probs = ordinal_probabilities(
        artifact["ordinal_models"],
        lgb_matrix,
    )

    weight = artifact["ordinal_weight"]
    blended_probs = (1 - weight) * cat_probs + weight * lgb_probs
    bias = np.asarray(artifact["log_bias"])

    return np.asarray(CLASSES)[
        np.argmax(
            np.log(np.clip(blended_probs, 1e-15, 1)) + bias,
            axis=1,
        )
    ]


def create_submission():
    """Memuat model, memprediksi test, dan menyimpan submission.csv."""

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            "Model belum tersedia. Jalankan python -m scripts.train terlebih dahulu."
        )

    test = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_PATH)
    validate_feature_frame(test)

    if not test[ID_COLUMN].equals(sample_submission[ID_COLUMN]):
        raise ValueError("Urutan building_id test dan sample submission berbeda.")

    artifact = joblib.load(MODEL_PATH)
    predictions = predict_ensemble(artifact, test)

    submission = sample_submission.copy()
    submission[TARGET] = grade_numbers(predictions)
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"Submission disimpan di: {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    create_submission()
