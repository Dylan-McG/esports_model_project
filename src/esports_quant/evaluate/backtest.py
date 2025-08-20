from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TypedDict

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
    train_patch_max: Optional[str]
    test_patch: Optional[str]


def _score_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold: int,
) -> FoldResult:
    X_train = train_df[["rating_diff", "bo"]].to_numpy(dtype=float)
    y_train = train_df["team_a_win"].to_numpy(dtype=int)
    X_test = test_df[["rating_diff", "bo"]].to_numpy(dtype=float)
    y_test = test_df["team_a_win"].to_numpy(dtype=int)

    model = make_baseline_pipeline()
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]

    train_patch_max: Optional[str] = None
    test_patch: Optional[str] = None
    if "patch" in train_df.columns:
        try:
            train_patch_max = str(train_df["patch"].dropna().astype(str).max())
        except Exception:
            train_patch_max = None
    if "patch" in test_df.columns:
        try:
            test_patch = str(test_df["patch"].dropna().astype(str).mode().iloc[0]) if not test_df.empty else None
        except Exception:
            test_patch = None

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
        "train_patch_max": train_patch_max,
        "test_patch": test_patch,
    }


def _rolling_by_rows(
    df: pd.DataFrame,
    n_splits: int,
    min_train: int,
    step: int,
) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    """Original rolling scheme by row counts."""
    folds: List[tuple[pd.DataFrame, pd.DataFrame]] = []
    i = min_train
    while i + step <= len(df) and len(folds) < n_splits:
        train_df = df.iloc[:i].copy()
        test_df = df.iloc[i : i + step].copy()
        folds.append((train_df, test_df))
        i += step
    return folds


def _rolling_by_patch(
    df: pd.DataFrame,
    datetime_col: str,
    patch_col: str,
    n_splits: int,
    min_train: int,
    embargo_days: int = 0,
) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Patch-aligned rolling:
      fold k trains on all rows strictly before patch P_k and tests on rows from patch P_k
      (optionally skipping the first `embargo_days` of that patch).
    """
    if patch_col not in df.columns:
        raise ValueError(f"snap_to_patch=True requires column '{patch_col}' in df")

    # ensure chronological order
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # identify contiguous patch segments in order
    patch_series = df[patch_col].astype(str).fillna("NA_PATCH")
    prev = patch_series.shift(1).fillna(patch_series.iloc[0])
    patch_change = patch_series.ne(prev)

    segment_starts = df.index[patch_change].tolist()
    segment_starts.append(len(df))  # sentinel end

    segments: list[tuple[int, int, str]] = []
    for s, e in zip(segment_starts[:-1], segment_starts[1:], strict=False):
        segments.append((int(s), int(e), str(patch_series.iloc[s])))

    folds: List[tuple[pd.DataFrame, pd.DataFrame]] = []

    # find first test segment where the training prefix meets min_train
    seg_ptr = 0
    while seg_ptr < len(segments) and len(df.iloc[: segments[seg_ptr][0]]) < min_train:
        seg_ptr += 1

    fold_idx = 0
    while seg_ptr < len(segments) and fold_idx < n_splits:
        test_s, test_e, _test_patch = segments[seg_ptr]

        # embargo: drop earliest days inside the test segment
        if embargo_days > 0:
            dt_cut = df.loc[test_s : test_e - 1, datetime_col].min() + pd.Timedelta(days=embargo_days)
            mask = df.index.to_series().between(test_s, test_e - 1)
            mask &= df[datetime_col] >= dt_cut
            test_df = df.loc[mask].copy()
        else:
            test_df = df.iloc[test_s:test_e].copy()

        train_df = df.iloc[:test_s].copy()

        if len(train_df) >= min_train and not test_df.empty:
            folds.append((train_df, test_df))
            fold_idx += 1

        seg_ptr += 1

    return folds


def rolling_backtest(
    df: pd.DataFrame,
    datetime_col: str,
    n_splits: int = 5,
    min_train: int = 2000,
    step: int = 500,
    out_csv: str | Path | None = None,
    out_plot: str | Path | None = None,
    *,
    snap_to_patch: bool = False,
    patch_col: str | None = "patch",
    embargo_days: int = 0,
) -> pd.DataFrame:
    """
    Chronological backtest.

    Modes:
      • Default (snap_to_patch=False): rolling by row counts (min_train/step).
      • Patch-aligned (snap_to_patch=True): each fold tests on the next patch as a block.
        Optionally skip the first `embargo_days` of the test patch.
    """
    # --- Label shim (accept 'radiant_win' as label) ---
    if "team_a_win" not in df.columns:
        if "radiant_win" in df.columns:
            df = df.rename(columns={"radiant_win": "team_a_win"})
        else:
            raise ValueError("Expected 'team_a_win' or 'radiant_win' column in df")

    # Sort chronologically
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # Build folds
    if not snap_to_patch:
        if len(df) < (min_train + step):
            raise ValueError("Not enough rows for requested min_train + step")
        fold_pairs = _rolling_by_rows(df, n_splits=n_splits, min_train=min_train, step=step)
    else:
        if patch_col is None:
            raise ValueError("snap_to_patch=True requires a non-None patch_col")
        fold_pairs = _rolling_by_patch(
            df,
            datetime_col=datetime_col,
            patch_col=patch_col,
            n_splits=n_splits,
            min_train=min_train,
            embargo_days=embargo_days,
        )

    # Score folds
    fold_results: List[FoldResult] = []
    for k, (train_df, test_df) in enumerate(fold_pairs):
        res = _score_fold(train_df, test_df, fold=k)
        fold_results.append(res)

    out = pd.DataFrame(fold_results)

    # --- ALWAYS write CSV if a path is provided ---
    if out_csv is not None:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)

    # --- ALWAYS write a plot if a path is provided (even if empty) ---
    if out_plot is not None:
        p = Path(out_plot)
        p.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        if out.empty:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No folds produced.\nCheck min_train/step/n_splits vs dataset size.",
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
            )
        else:
            ax.plot(out["fold"], out["log_loss"], marker="o", label="log_loss")
            ax.plot(out["fold"], out["brier"], marker="o", label="brier")
            ax.plot(out["fold"], out["ece"], marker="o", label="ece")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Score (lower is better)")
            title = "Patch-aligned backtest metrics" if snap_to_patch else "Rolling backtest metrics"
            ax.set_title(title)
            ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)

    return out
