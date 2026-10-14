"""Feature engineering untuk model CatBoost dan LightGBM."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


TARGET = "damage_grade"
ID_COLUMN = "building_id"
CLASSES = tuple(f"Grade {number}" for number in range(1, 6))
DEFAULT_PREFIX_LENGTHS = (3, 4, 5, 6, 7, 8)

RAW_CATEGORICAL_COLUMNS = (
    "district_id",
    "vdcmun_id",
    "ward_id",
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "superstructure",
)

REQUIRED_FEATURE_COLUMNS = (
    ID_COLUMN,
    "district_id",
    "vdcmun_id",
    "ward_id",
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "height_ft_pre_eq",
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "superstructure",
)


def validate_feature_frame(data):
    """Memastikan semua fitur yang dibutuhkan tersedia."""

    missing = sorted(set(REQUIRED_FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {missing}")


def clean_building_id(values):
    """Membersihkan building ID sebelum mengambil bagian prefix."""

    return (
        values.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(12)
    )


class DamageFeatureEngineer(BaseEstimator, TransformerMixin):
    """Membuat fitur yang digunakan oleh CatBoost dan LightGBM."""

    def __init__(self, prefix_lengths=DEFAULT_PREFIX_LENGTHS):
        self.prefix_lengths = prefix_lengths

    @property
    def categorical_columns(self):
        """Mengembalikan nama seluruh fitur kategorikal."""

        prefix_columns = [
            f"building_prefix_{length}" for length in self.prefix_lengths
        ]
        return list(RAW_CATEGORICAL_COLUMNS) + prefix_columns

    @property
    def numeric_columns(self):
        """Mengembalikan nama seluruh fitur numerik."""

        return [
            "count_floors_pre_eq",
            "plinth_area_sq_ft",
            "height_ft_pre_eq",
            "age_is_999",
            "age_clean",
            "floor_height_ratio",
            "area_per_floor",
            "volume_proxy",
        ]

    def fit(self, X, y=None):
        """Memeriksa kolom input dan mengembalikan transformer."""

        validate_feature_frame(X)
        return self

    def transform(self, X):
        """Mengubah data mentah menjadi fitur untuk modeling."""

        validate_feature_frame(X)
        data = X.loc[:, REQUIRED_FEATURE_COLUMNS].copy()

        building_id = clean_building_id(data.pop(ID_COLUMN))
        for length in self.prefix_lengths:
            data[f"building_prefix_{length}"] = building_id.str[:length]

        age = pd.to_numeric(data.pop("age_building"), errors="coerce")
        data["age_is_999"] = age.eq(999).astype("int8")
        data["age_clean"] = age.clip(0, 100)

        floors = pd.to_numeric(
            data["count_floors_pre_eq"], errors="coerce"
        ).clip(lower=1)
        height = pd.to_numeric(data["height_ft_pre_eq"], errors="coerce")
        area = pd.to_numeric(data["plinth_area_sq_ft"], errors="coerce")

        data["floor_height_ratio"] = height / floors
        data["area_per_floor"] = area / floors
        data["volume_proxy"] = area * height

        for column in self.categorical_columns:
            data[column] = data[column].astype("string").fillna("__MISSING__")

        for column in self.numeric_columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        return data


def location_keys(X, prefix_lengths=DEFAULT_PREFIX_LENGTHS):
    """Membuat kelompok lokasi dan material untuk target encoding."""

    keys = pd.DataFrame(index=X.index)
    for column in ["district_id", "vdcmun_id", "ward_id"]:
        keys[column] = X[column].astype("string").fillna("__MISSING__")

    building_id = clean_building_id(X[ID_COLUMN])
    for length in prefix_lengths:
        if length >= 4:
            keys[f"prefix_{length}"] = building_id.str[:length]

    material = X["superstructure"].astype("string").fillna("__MISSING__")
    foundation = X["foundation_type"].astype("string").fillna("__MISSING__")
    keys["material_foundation"] = material + "|" + foundation
    keys["ward_material"] = keys["ward_id"] + "|" + material
    keys["prefix6_material"] = building_id.str[:6] + "|" + material
    return keys


def fit_target_maps(keys, y_number, smoothing=40.0):
    """Menghitung mapping target encoding dengan smoothing."""

    priors = np.bincount(y_number, minlength=6)[1:].astype(float)
    priors = priors / priors.sum()
    severity_prior = float(np.mean(y_number))
    maps = {}

    for column in keys.columns:
        temporary = pd.DataFrame({"key": keys[column], "target": y_number})
        counts = temporary.groupby("key")["target"].size()
        severity = temporary.groupby("key")["target"].mean()
        denominator = counts + smoothing

        column_map = {
            "count": counts,
            "severity": (
                severity * counts + severity_prior * smoothing
            ) / denominator,
        }

        for grade in range(1, 6):
            positive = temporary["target"].eq(grade).groupby(
                temporary["key"]
            ).sum()
            column_map[f"p{grade}"] = (
                positive + priors[grade - 1] * smoothing
            ) / denominator

        maps[column] = column_map

    maps["__global__"] = {
        "priors": priors,
        "severity": severity_prior,
    }
    return maps


def transform_target_encoding(keys, maps):
    """Mengubah kelompok fitur menjadi fitur target encoding."""

    priors = np.asarray(maps["__global__"]["priors"])
    severity_prior = maps["__global__"]["severity"]
    result = {}

    for column in keys.columns:
        column_map = maps[column]
        result[f"te_{column}_log_count"] = np.log1p(
            keys[column].map(column_map["count"]).fillna(0)
        )
        result[f"te_{column}_severity"] = keys[column].map(
            column_map["severity"]
        ).fillna(severity_prior)

        for grade in range(1, 6):
            result[f"te_{column}_p{grade}"] = keys[column].map(
                column_map[f"p{grade}"]
            ).fillna(priors[grade - 1])

    return pd.DataFrame(result, index=keys.index, dtype=np.float32)


def oof_target_encoding(
    X,
    y_number,
    n_splits=3,
    smoothing=40.0,
    random_state=42,
    prefix_lengths=DEFAULT_PREFIX_LENGTHS,
):
    """Membuat out-of-fold target encoding untuk mencegah leakage."""

    keys = location_keys(X, prefix_lengths).reset_index(drop=True)
    encoded = None
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    for train_index, valid_index in splitter.split(keys, y_number):
        fold_maps = fit_target_maps(
            keys.iloc[train_index],
            y_number[train_index],
            smoothing,
        )
        fold_data = transform_target_encoding(
            keys.iloc[valid_index],
            fold_maps,
        )

        if encoded is None:
            encoded = pd.DataFrame(
                np.nan,
                index=np.arange(len(X)),
                columns=fold_data.columns,
                dtype=np.float32,
            )
        encoded.iloc[valid_index] = fold_data.to_numpy()

    full_maps = fit_target_maps(keys, y_number, smoothing)
    return encoded, full_maps


def raw_lgb_features_fit(X, prefix_lengths=DEFAULT_PREFIX_LENGTHS):
    """Mengubah fitur kategorikal menjadi angka untuk training LightGBM."""

    data = DamageFeatureEngineer(prefix_lengths).fit_transform(X)
    data = data.reset_index(drop=True)
    category_maps = {}

    for column in data.columns:
        if column in DamageFeatureEngineer(prefix_lengths).categorical_columns:
            values = data[column].astype("string").fillna("__MISSING__")
            categories = sorted(values.unique())
            mapping = {value: index for index, value in enumerate(categories)}
            category_maps[column] = mapping
            data[column] = values.map(mapping).fillna(-1).astype("int32")
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce").astype(
                "float32"
            )

    return data, category_maps


def raw_lgb_features_transform(
    X,
    category_maps,
    prefix_lengths=DEFAULT_PREFIX_LENGTHS,
):
    """Menerapkan mapping kategori training pada data baru."""

    data = DamageFeatureEngineer(prefix_lengths).transform(X)
    data = data.reset_index(drop=True)

    for column in data.columns:
        if column in category_maps:
            values = data[column].astype("string").fillna("__MISSING__")
            data[column] = values.map(category_maps[column]).fillna(-1).astype(
                "int32"
            )
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce").astype(
                "float32"
            )

    return data


def make_lgb_matrix(raw_features, encoded_features):
    """Menggabungkan fitur biasa dan target encoding untuk LightGBM."""

    result = pd.concat(
        [raw_features.reset_index(drop=True), encoded_features.reset_index(drop=True)],
        axis=1,
    )
    return result.replace([np.inf, -np.inf], np.nan)
