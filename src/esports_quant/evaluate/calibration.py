from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline

from ..models.baseline import make_baseline_pipeline
from ..utils.io import save_pickle
from .metrics import expected_calibration_error


def _split_for_calibration(
    df: pd.DataFrame, random_state: int = 2025
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Deterministic 80/20 split (chronological keeps our leakage risk lower)
    df = df.sort_values("start_time").reset_index(drop=True)
    split = int(len(df) * 0.8)

    feats = ["rating_diff", "bo"]  # minimal baseline features
    X_train = df.loc[: split - 1, feats].to_numpy(dtype=float)
    y_train = df.loc[: split - 1, "team_a_win"].to_numpy(dtype=int)

    X_cal = df.loc[split:, feats].to_numpy(dtype=float)
    y_cal = df.loc[split:, "team_a_win"].to_numpy(dtype=int)
    return X_train, X_cal, y_train, y_cal


def _prefit_base_then_calibrate(
    base_model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str,  # "platt" | "isotonic"
):
    base_model.fit(X_train, y_train)

    # Pre-calibration metrics (on calibration split)
    y_prob_pre = base_model.predict_proba(X_cal)[:, 1]
    pre = {
        "log_loss": float(
            max(
                0.0,
                -np.mean(
                    y_cal * np.log(np.clip(y_prob_pre, 1e-12, 1 - 1e-12))
                    + (1 - y_cal) * np.log(np.clip(1 - y_prob_pre, 1e-12, 1 - 1e-12))
                ),
            )
        ),
        "brier": float(np.mean((y_prob_pre - y_cal) ** 2)),
        "ece": float(expected_calibration_error(y_cal, y_prob_pre)),
    }

    skl_method = "sigmoid" if method == "platt" else "isotonic"
    # sklearn >= 1.6 uses "estimator", not "base_estimator"
    calibrator = CalibratedClassifierCV(estimator=base_model, method=skl_method, cv="prefit")
    calibrator.fit(X_cal, y_cal)

    # Post-calibration metrics
    y_prob_post = calibrator.predict_proba(X_cal)[:, 1]
    post = {
        "log_loss": float(
            max(
                0.0,
                -np.mean(
                    y_cal * np.log(np.clip(y_prob_post, 1e-12, 1 - 1e-12))
                    + (1 - y_cal) * np.log(np.clip(1 - y_prob_post, 1e-12, 1 - 1e-12))
                ),
            )
        ),
        "brier": float(np.mean((y_prob_post - y_cal) ** 2)),
        "ece": float(expected_calibration_error(y_cal, y_prob_post)),
    }

    return calibrator, pre, post


def _plot_reliability(
    y_true: np.ndarray,
    y_prob_pre: np.ndarray,
    y_prob_post: np.ndarray,
    out_path: str | Path,
) -> None:
    frac_pre, mean_pre = calibration_curve(y_true, y_prob_pre, n_bins=10, strategy="uniform")
    frac_post, mean_post = calibration_curve(y_true, y_prob_post, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfect", alpha=0.7)
    ax.plot(mean_pre, frac_pre, marker="o", linewidth=1.5, label="Pre-calibration")
    ax.plot(mean_post, frac_post, marker="o", linewidth=1.5, label="Post-calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibrate_and_save(
    df: pd.DataFrame,
    method: str,  # "platt" | "isotonic"
    out_model: str | Path,
    out_plot: str | Path | None = None,
    out_metrics: str | Path | None = None,
) -> dict[str, float]:
    base = make_baseline_pipeline()

    X_train, X_cal, y_train, y_cal = _split_for_calibration(df, random_state=2025)
    calibrator, pre, post = _prefit_base_then_calibrate(base, X_train, y_train, X_cal, y_cal, method)

    # Optional plot (wrapped lines to satisfy E501)
    if out_plot is not None:
        _plot_reliability(
            y_cal,
            base.predict_proba(X_cal)[:, 1],
            calibrator.predict_proba(X_cal)[:, 1],
            out_plot,
        )

    # Save calibrated model
    save_pickle(calibrator, out_model)

    if out_metrics is not None:
        import json

        Path(out_metrics).parent.mkdir(parents=True, exist_ok=True)
        with open(out_metrics, "w", encoding="utf-8") as f:
            json.dump(post, f, indent=2)

    return post
