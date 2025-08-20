# esports_model_project\src\esports_quant\models\baseline.py

# ruff: noqa: D401
"""
Baseline model definitions and training helpers.

Uses all numeric features (except IDs/timestamps/targets), imputes missing
values to 0.0, then scales and fits LogisticRegression. Saves the feature
list used at train time as `feature_names_` on the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline_pipeline(random_state: int = 2025) -> Pipeline:
    """
    Simple, robust baseline:
      SimpleImputer(0.0) -> StandardScaler -> LogisticRegression
    """
    clf = LogisticRegression(max_iter=200, random_state=random_state)
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipe


# ---- Feature selection helpers ------------------------------------------------
_DEFAULT_DROP: set[str] = {
    # IDs / timestamps / obvious non-features
    "match_id",
    "start_time",
    "start_time_unix",
    # targets (if present alongside team_a_win)
    "target",
    "team_a_win",
}


def _select_feature_columns(df: pd.DataFrame, extra_drop: Iterable[str] | None = None) -> list[str]:
    """
    Pick all numeric columns except known non-features and any user-specified drops.
    """
    drop = set(_DEFAULT_DROP)
    if extra_drop:
        drop |= set(extra_drop)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    feat_cols = [c for c in numeric_cols if c not in drop]
    return feat_cols


def _split_xy_dynamic(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Select X from all numeric columns (minus drops) and y from `team_a_win`.
    """
    if "team_a_win" not in df.columns:
        raise ValueError("Expected 'team_a_win' column in training DataFrame.")

    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    X = df[feature_cols].to_numpy(dtype=float)  # may contain NaNs; imputer will handle
    y = df["team_a_win"].to_numpy(dtype=int)
    return X, y, feature_cols


# ---- Training ----------------------------------------------------------------
# src/esports_quant/models/baseline.py


def train_baseline(
    df: pd.DataFrame,
    artifacts_dir: str | Path = "artifacts",
    model_name: str = "baseline_logit.pkl",
) -> Path:
    """
    Fit baseline pipeline on all rows and persist to artifacts.
    Returns the path to the saved model.

    - X: all numeric cols except IDs/timestamps/targets
    - y: `team_a_win`
    - The fitted pipeline gets `feature_names_` attached for consistent inference.
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols = _split_xy_dynamic(df)

    model = make_baseline_pipeline()
    model.fit(X, y)

    # Dynamic attribute; OK at runtime; avoids Pylance complaints.
    model.feature_names_ = feature_cols

    out_path = artifacts_path / model_name
    joblib.dump(model, out_path)
    return out_path
