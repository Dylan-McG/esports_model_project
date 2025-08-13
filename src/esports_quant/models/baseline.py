# ruff: noqa: D401
"""
Baseline model definitions and training helpers.

We expose a reusable `make_baseline_pipeline()` that the rest of the pipeline
(calibration, backtests, CLI) can import without duplication.

Features expected (by column name):
    - "rating_diff" : float
    - "bo"          : int (1/3/5)
Target:
    - "team_a_win"  : 0/1
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline_pipeline(random_state: int = 2025) -> Pipeline:
    """
    Build a simple, robust baseline:
        StandardScaler -> LogisticRegression

    We scale both numerical features ("rating_diff", "bo"). LR is well-behaved,
    fast to train, and provides calibrated-ish probabilities once calibrated
    downstream.
    """
    clf = LogisticRegression(max_iter=200, random_state=random_state)
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipe


def _split_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[["rating_diff", "bo"]].to_numpy(dtype=float)
    y = df["team_a_win"].to_numpy(dtype=int)
    return x, y


def train_baseline(
    df: pd.DataFrame,
    artifacts_dir: str | Path = "artifacts",
    model_name: str = "baseline_logit.pkl",
) -> Path:
    """
    Fit baseline pipeline on all rows and persist to artifacts.
    Returns the path to the saved model.
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    x, y = _split_xy(df)
    model = make_baseline_pipeline()
    model.fit(x, y)

    out_path = artifacts_path / model_name
    joblib.dump(model, out_path)
    return out_path
