# Purpose: Minimal logistic baseline on simple features.

from __future__ import annotations  # typing future

from dataclasses import dataclass  # result container
from pathlib import Path  # paths

import pandas as pd  # dataframes
from sklearn.linear_model import LogisticRegression  # classifier
from sklearn.model_selection import train_test_split  # split
from sklearn.pipeline import Pipeline  # compose steps
from sklearn.preprocessing import StandardScaler  # scaling

from ..evaluate.metrics import compute_core_metrics  # metrics
from ..utils.io import save_pickle  # persist model


@dataclass
class TrainResult:
    model_path: Path  # saved artifact
    metrics: dict[str, float]  # validation metrics


def train_baseline(
    df: pd.DataFrame, artifacts_dir: str | Path = "artifacts", seed: int = 7
) -> TrainResult:
    feature_cols = ["rating_diff", "bo"]  # features
    X = df[feature_cols].values  # feature matrix
    y = df["team_a_win"].values  # labels
    X_tr, X_va, y_tr, y_va = train_test_split(  # split
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    pipe = Pipeline(  # scaler + logistic regression
        steps=[
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )
    pipe.fit(X_tr, y_tr)  # train
    y_prob = pipe.predict_proba(X_va)[:, 1]  # val probs
    metrics = compute_core_metrics(y_va, y_prob)  # eval
    model_dir = Path(artifacts_dir)  # ensure dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_logit.pkl"  # artifact path
    save_pickle(pipe, model_path)  # save model
    return TrainResult(model_path=model_path, metrics=metrics)  # result
