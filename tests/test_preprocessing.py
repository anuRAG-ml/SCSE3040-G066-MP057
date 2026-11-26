import numpy as np
import pandas as pd

from src.mp057.preprocessing import (
    normalize_fabric,
    prepare_features,
    build_preprocessor,
)


def test_normalize_fabric():
    values = pd.Series([" SILK ", "Sil.", "POL.", "Cot."])
    result = normalize_fabric(values)

    assert result.tolist() == ["silk", "silk", "polyester", "cotton"]


def test_prepare_features():
    df = pd.DataFrame({
        "loom_age_yr": [10, 20],
        "thread_count": [-1, 100],
        "power_cuts_min": [20, 30],
        "weaver_exp_yr": [5, 10],
        "humidity_pct": [50, 60],
        "shift_hours": [8, 9],
        "fabric": ["Silk", "Cot."],
        "shed": ["Shed 1", "Shed 2"],
        "metres_checked_at_folding": [40, 50],
    })

    X = prepare_features(df)

    assert X.shape == (2, 8)
    assert pd.isna(X.loc[0, "thread_count"])
    assert "metres_checked_at_folding" not in X.columns


def test_preprocessor():
    df = pd.DataFrame({
        "loom_age_yr": [10, 20, 15],
        "thread_count": [-1, 100, 120],
        "power_cuts_min": [20, 30, 40],
        "weaver_exp_yr": [5, 10, 15],
        "humidity_pct": [50, 60, 70],
        "shift_hours": [8, 9, 10],
        "fabric": ["Silk", "Cot.", "POL."],
        "shed": ["Shed 1", "Shed 2", "Shed 3"],
    })

    X = prepare_features(df)
    transformed = build_preprocessor().fit_transform(X)

    assert transformed.shape == (3, 12)
    assert np.isfinite(transformed).all()
