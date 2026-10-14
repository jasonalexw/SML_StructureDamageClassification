"""Training ensemble CatBoost dan LightGBM ordinal."""

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .feature_engineering import (
    CLASSES,
    DEFAULT_PREFIX_LENGTHS,
    TARGET,
    DamageFeatureEngineer,
    location_keys,
    make_lgb_matrix,
    oof_target_encoding,
    raw_lgb_features_fit,
    raw_lgb_features_transform,
    transform_target_encoding,
)
from .preprocess import (
    audit_data,
    build_column_transformer,
    calculate_class_weights,
    grade_numbers,
    load_data,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = ROOT_DIR / "models" / "pipeline_model_ensemble.pkl"
METADATA_PATH = ROOT_DIR / "models" / "metadata.json"


@dataclass
class RunConfig:
    """Konfigurasi utama yang digunakan saat training."""

    seed: int = 42
    validation_size: float = 0.20
    selection_size: int = 140_000
    cat_search_iterations: int = 160
    cat_final_iterations: int = 600
    lgb_iterations: int = 500
    target_encoding_folds: int = 3
    target_encoding_smoothing: float = 40.0
    depth: int = 7
    cat_learning_rate: float = 0.12
    class_weight_exponent: float = 0.25
    threads: int = max(1, (os.cpu_count() or 2) - 1)
    prefix_lengths: tuple = DEFAULT_PREFIX_LENGTHS


def build_catboost(config, y, iterations, seed):
    """Membuat Pipeline feature engineering, ColumnTransformer, dan CatBoost."""

    features = DamageFeatureEngineer(config.prefix_lengths)
    preprocessor = build_column_transformer(config.prefix_lengths)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=iterations,
        depth=config.depth,
        learning_rate=config.cat_learning_rate,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        cat_features=features.categorical_columns,
        class_weights=calculate_class_weights(
            y, config.class_weight_exponent
        ),
        random_seed=seed,
        thread_count=config.threads,
        allow_writing_files=False,
        verbose=False,
    )

    return Pipeline(
        [
            ("feature_engineering", features),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def catboost_probabilities(model, X):
    """Mengurutkan probabilitas CatBoost sesuai Grade 1 sampai Grade 5."""

    probabilities = model.predict_proba(X)
    model_classes = list(model.named_steps["model"].classes_)
    positions = [model_classes.index(label) for label in CLASSES]
    return probabilities[:, positions]


def build_ordinal_model(config, threshold, iterations):
    """Membuat satu binary LightGBM untuk batas grade tertentu."""

    return LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        learning_rate=0.045,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=6.0,
        random_state=config.seed + threshold * 101,
        n_jobs=config.threads,
        verbosity=-1,
    )


def fit_ordinal_models(
    X_train,
    y_train,
    config,
    X_valid=None,
    y_valid=None,
):
    """Melatih empat LightGBM untuk target y>1 sampai y>4."""

    models = []

    for threshold in range(1, 5):
        model = build_ordinal_model(
            config,
            threshold,
            config.lgb_iterations,
        )
        binary_target = (y_train > threshold).astype("int8")
        counts = np.bincount(binary_target, minlength=2).astype(float)
        sample_weight = np.where(
            binary_target == 1,
            (counts[0] / max(counts[1], 1)) ** 0.15,
            1.0,
        )

        fit_options = {
            "sample_weight": sample_weight,
            "callbacks": [log_evaluation(period=0)],
        }

        if X_valid is not None and y_valid is not None:
            fit_options["eval_set"] = [
                (X_valid, (y_valid > threshold).astype("int8"))
            ]
            fit_options["callbacks"] = [
                early_stopping(60, verbose=False),
                log_evaluation(period=0),
            ]

        model.fit(X_train, binary_target, **fit_options)
        models.append(model)
        print(f"LightGBM untuk y > {threshold} selesai.")

    return models


def ordinal_probabilities(models, X):
    """Mengubah probabilitas kumulatif menjadi probabilitas lima grade."""

    cumulative = np.column_stack(
        [model.predict_proba(X)[:, 1] for model in models]
    )
    cumulative = np.minimum.accumulate(cumulative, axis=1)

    probabilities = np.column_stack(
        [
            1 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1] - cumulative[:, 2],
            cumulative[:, 2] - cumulative[:, 3],
            cumulative[:, 3],
        ]
    )
    probabilities = np.clip(probabilities, 1e-10, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def find_best_blend(cat_probs, lgb_probs, y_valid):
    """Mencari bobot blend dan bias kelas dengan macro-F1 terbaik."""

    labels = np.asarray(CLASSES)
    best_weight = 0.0
    best_score = -1.0
    best_probabilities = cat_probs

    for weight in np.arange(0.10, 0.61, 0.05):
        blended = (1 - weight) * cat_probs + weight * lgb_probs
        predictions = labels[np.argmax(blended, axis=1)]
        score = f1_score(y_valid, predictions, average="macro")

        if score > best_score:
            best_weight = float(weight)
            best_score = float(score)
            best_probabilities = blended

    log_probabilities = np.log(np.clip(best_probabilities, 1e-15, 1))
    bias = np.zeros(5)

    def bias_score(candidate):
        predictions = labels[np.argmax(log_probabilities + candidate, axis=1)]
        return f1_score(y_valid, predictions, average="macro")

    calibrated_score = bias_score(bias)
    for step in [0.16, 0.08, 0.04, 0.02]:
        for _ in range(3):
            improved = False
            for class_index in range(5):
                for direction in [-1, 1]:
                    candidate = bias.copy()
                    candidate[class_index] += direction * step
                    candidate -= candidate.mean()
                    score = bias_score(candidate)
                    if score > calibrated_score:
                        bias = candidate
                        calibrated_score = float(score)
                        improved = True
            if not improved:
                break

    return best_weight, bias, best_score, calibrated_score


def select_ensemble(X, y, config):
    """Memilih bobot ensemble menggunakan sebagian data training."""

    all_indices = np.arange(len(y))
    if config.selection_size < len(y):
        selected_indices, _ = train_test_split(
            all_indices,
            train_size=config.selection_size,
            stratify=y,
            random_state=config.seed,
        )
    else:
        selected_indices = all_indices

    train_index, valid_index = train_test_split(
        selected_indices,
        test_size=config.validation_size,
        stratify=y.iloc[selected_indices],
        random_state=config.seed,
    )

    X_train = X.iloc[train_index].reset_index(drop=True)
    X_valid = X.iloc[valid_index].reset_index(drop=True)
    y_train = y.iloc[train_index].reset_index(drop=True)
    y_valid = y.iloc[valid_index].reset_index(drop=True)

    print(f"Seleksi model: {len(X_train)} train, {len(X_valid)} validation")

    catboost = build_catboost(
        config,
        y_train,
        config.cat_search_iterations,
        config.seed,
    )
    catboost.fit(X_train, y_train)
    cat_probs = catboost_probabilities(catboost, X_valid)

    raw_train, category_maps = raw_lgb_features_fit(
        X_train, config.prefix_lengths
    )
    raw_valid = raw_lgb_features_transform(
        X_valid, category_maps, config.prefix_lengths
    )
    y_train_number = grade_numbers(y_train)
    y_valid_number = grade_numbers(y_valid)

    encoded_train, target_maps = oof_target_encoding(
        X_train,
        y_train_number,
        config.target_encoding_folds,
        config.target_encoding_smoothing,
        config.seed + 91,
        config.prefix_lengths,
    )
    encoded_valid = transform_target_encoding(
        location_keys(X_valid, config.prefix_lengths),
        target_maps,
    )

    lgb_train = make_lgb_matrix(raw_train, encoded_train)
    lgb_valid = make_lgb_matrix(raw_valid, encoded_valid)
    ordinal_models = fit_ordinal_models(
        lgb_train,
        y_train_number,
        config,
        lgb_valid,
        y_valid_number,
    )
    lgb_probs = ordinal_probabilities(ordinal_models, lgb_valid)

    weight, bias, raw_score, final_score = find_best_blend(
        cat_probs,
        lgb_probs,
        y_valid,
    )

    blended = (1 - weight) * cat_probs + weight * lgb_probs
    predictions = np.asarray(CLASSES)[
        np.argmax(np.log(np.clip(blended, 1e-15, 1)) + bias, axis=1)
    ]

    print(f"Bobot LightGBM: {weight:.2f}")
    print(f"Macro-F1 sebelum bias: {raw_score:.4f}")
    print(f"Macro-F1 setelah bias: {final_score:.4f}")
    print(classification_report(y_valid, predictions, zero_division=0))

    del catboost, ordinal_models, lgb_train, lgb_valid
    gc.collect()

    return {
        "ordinal_weight": weight,
        "log_bias": bias.tolist(),
        "raw_macro_f1": raw_score,
        "calibrated_macro_f1": final_score,
    }


def train_model():
    """Menjalankan seleksi ensemble lalu melatih model pada seluruh data."""

    start_time = time.perf_counter()
    config = RunConfig()
    train, test = load_data(DATA_DIR)
    audit_data(train, test)

    X = train.drop(columns=TARGET)
    y = train[TARGET].astype(str)
    selection = select_ensemble(X, y, config)

    print("Training CatBoost final...")
    catboost = build_catboost(
        config,
        y,
        config.cat_final_iterations,
        config.seed + 5000,
    )
    catboost.fit(X, y)

    print("Membuat fitur LightGBM final...")
    raw_train, category_maps = raw_lgb_features_fit(X, config.prefix_lengths)
    y_number = grade_numbers(y)
    encoded_train, target_maps = oof_target_encoding(
        X,
        y_number,
        config.target_encoding_folds,
        config.target_encoding_smoothing,
        config.seed + 91,
        config.prefix_lengths,
    )
    lgb_train = make_lgb_matrix(raw_train, encoded_train)
    ordinal_models = fit_ordinal_models(lgb_train, y_number, config)

    artifact = {
        "classes": list(CLASSES),
        "catboost_pipeline": catboost,
        "ordinal_models": ordinal_models,
        "ordinal_weight": selection["ordinal_weight"],
        "log_bias": selection["log_bias"],
        "category_maps": category_maps,
        "target_maps": target_maps,
        "prefix_lengths": list(config.prefix_lengths),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)

    metadata = {
        "config": asdict(config),
        "selection": selection,
        "training_rows": len(train),
        "training_seconds": time.perf_counter() - start_time,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Model disimpan di: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
