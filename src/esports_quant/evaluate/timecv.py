# src/esports_quant/evaluate/timecv.py

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss


def block_time_cv(
    df: pd.DataFrame,
    datetime_col: str,
    feature_cols: list[str],
    target_col: str,
    model: Pipeline,
    n_splits: int = 5,
) -> dict[str, float]:
    df = df.sort_values(datetime_col).reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_splits + 1)  # first block = train0, then test1, etc.

    ll_list, br_list = [], []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]

        Xtr, ytr = train[feature_cols].to_numpy(), train[target_col].to_numpy()
        Xte, yte = test[feature_cols].to_numpy(), test[target_col].to_numpy()

        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        ll_list.append(log_loss(yte, p))
        br_list.append(brier_score_loss(yte, p))

    return {
        "cv_log_loss_mean": float(np.mean(ll_list)),
        "cv_brier_mean": float(np.mean(br_list)),
        "cv_log_loss_std": float(np.std(ll_list)),
        "cv_brier_std": float(np.std(br_list)),
        "folds": n_splits,
    }
