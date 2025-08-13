from __future__ import annotations

from pathlib import Path
from typing import List, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..models.baseline import make_baseline_pipeline
from .metrics import expected_calibration_error


class FoldResult(TypedDict):
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    log_loss: float
    brier: float
    ece: float


def _score_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, fold: int) -> FoldResult:
    X_train = train_df[["rating_diff", "bo"]].to_numpy(dtype=float)
    y_train = train_df["team_a_win"].to_numpy(dtype=int)
    X_test = test_df[["rating_diff", "bo"]].to_numpy(dtype=float)
    y_test = test_df["team_a_win"].to_numpy(dtype=int)

    model = make_baseline_pipeline()
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]

    return {
        "fold": fold,
        "train_start": int(train_df.index.min()),
        "train_end": int(train_df.index.max()),
        "test_start": int(test_df.index.min()),
        "test_end": int(test_df.index.max()),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "log_loss": float(log_loss(y_test, prob)),
        "brier": float(brier_score_loss(y_test, prob)),
        "ece": float(expected_calibration_error(y_test, prob, n_bins=10)),
    }


def rolling_backtest(
    df: pd.DataFrame,
    datetime_col: str,
    n_splits: int = 5,
    min_train: int = 2000,
    step: int = 500,
    out_csv: str | Path | None = None,
    out_plot: str | Path | None = None,
) -> pd.DataFrame:
    """
    Chronological rolling-origin backtest. At each step, train on [0:i) and
    test on [i:i+step). Returns a DataFrame of per-fold metrics and optionally
    writes a CSV and a quick line plot.
    """
    df = df.sort_values(datetime_col).reset_index(drop=True)

    if len(df) < (min_train + step):
        raise ValueError("Not enough rows for requested min_train + step")

    results: List[FoldResult] = []
    fold = 0
    i = min_train
    while i + step <= len(df) and fold < n_splits:
        train_df = df.iloc[:i].copy()
        test_df = df.iloc[i : i + step].copy()
        res = _score_fold(train_df, test_df, fold=fold)
        results.append(res)
        fold += 1
        i += step

    out = pd.DataFrame(results)

    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    if out_plot is not None and not out.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(out["fold"], out["log_loss"], marker="o", label="log_loss")
        ax.plot(out["fold"], out["brier"], marker="o", label="brier")
        ax.plot(out["fold"], out["ece"], marker="o", label="ece")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Score (lower is better)")
        ax.set_title("Rolling backtest metrics")
        ax.legend(loc="best")
        fig.tight_layout()
        Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=150)
        plt.close(fig)

    return out
