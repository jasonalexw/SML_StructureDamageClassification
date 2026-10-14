"""Validasi data dan preprocessing untuk model klasifikasi."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .feature_engineering import (
    CLASSES,
    ID_COLUMN,
    TARGET,
    DamageFeatureEngineer,
    validate_feature_frame,
)


def load_data(data_dir):
    """Membaca train.csv dan test.csv dari folder data."""

    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def audit_data(train, test):
    """Memeriksa kolom, target, dan keunikan building ID."""

    validate_feature_frame(train)
    validate_feature_frame(test)

    if TARGET not in train.columns:
        raise ValueError(f"Target {TARGET} tidak ditemukan pada train.csv")
    if train[TARGET].isna().any():
        raise ValueError("Target masih memiliki missing value.")

    unknown_labels = set(train[TARGET].unique()) - set(CLASSES)
    if unknown_labels:
        raise ValueError(f"Label target tidak dikenal: {unknown_labels}")
    if train[ID_COLUMN].duplicated().any():
        raise ValueError("building_id pada train tidak unik.")
    if test[ID_COLUMN].duplicated().any():
        raise ValueError("building_id pada test tidak unik.")

    print("Ukuran train:", train.shape)
    print("Ukuran test :", test.shape)
    print("\nDistribusi target:")
    print(train[TARGET].value_counts().sort_index())


def grade_numbers(labels):
    """Mengubah Grade 1 sampai Grade 5 menjadi angka 1 sampai 5."""

    numbers = pd.Series(labels, dtype="string").str.extract(
        r"(\d+)$", expand=False
    )
    if numbers.isna().any():
        raise ValueError("Format damage_grade harus Grade 1 sampai Grade 5.")
    return numbers.astype("int8").to_numpy()


def calculate_class_weights(y, exponent=0.25):
    """Menghitung bobot kelas sederhana untuk data tidak seimbang."""

    counts = y.value_counts()
    return {
        label: float((len(y) / (len(counts) * count)) ** exponent)
        for label, count in counts.items()
    }


def build_column_transformer(prefix_lengths=(3, 4, 5, 6, 7, 8)):
    """Membuat ColumnTransformer untuk pipeline CatBoost.

    Fitur kategorikal dan numerik hanya diimputasi karena CatBoost tidak
    membutuhkan one-hot encoding atau standardisasi.
    """

    feature_engineer = DamageFeatureEngineer(prefix_lengths)

    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent"))]
    )
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )

    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                feature_engineer.categorical_columns,
            ),
            ("numeric", numeric_pipeline, feature_engineer.numeric_columns),
        ],
        verbose_feature_names_out=False,
    )

    return transformer.set_output(transform="pandas")
