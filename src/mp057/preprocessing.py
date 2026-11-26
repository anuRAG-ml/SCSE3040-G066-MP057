import pandas as pd


FEATURE_COLUMNS = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
    "fabric",
    "shed",
]


def normalize_fabric(series):
    """Normalize fabric names and spelling variants."""
    mapping = {
        "silk": "silk",
        "sil.": "silk",
        "polyester": "polyester",
        "pol.": "polyester",
        "cotton": "cotton",
        "cot.": "cotton",
    }

    return series.astype(str).str.strip().str.lower().map(mapping)


def prepare_features(df):
    """Clean and select model input features."""
    X = df[FEATURE_COLUMNS].copy()

    X["fabric"] = normalize_fabric(X["fabric"])
    X["thread_count"] = X["thread_count"].replace(-1, float("nan"))

    return X

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
]

CATEGORICAL_FEATURES = ["fabric", "shed"]


def build_preprocessor():
    """Build preprocessing transformer for numerical and categorical data."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor
