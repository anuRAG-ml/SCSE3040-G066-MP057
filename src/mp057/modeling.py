from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.mp057.preprocessing import build_preprocessor


def build_model_pipeline(random_state=42):
    """Build complete preprocessing and regression pipeline."""

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

    return pipeline

from sklearn.ensemble import GradientBoostingRegressor


def build_gradient_boosting_pipeline(random_state=42):
    """Build preprocessing and Gradient Boosting regression pipeline."""

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

    return pipeline
