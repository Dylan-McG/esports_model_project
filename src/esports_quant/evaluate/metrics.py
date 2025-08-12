# Purpose: Core evaluation metrics (log loss, Brier, ECE).

from __future__ import annotations  # typing future

import numpy as np  # arrays
from sklearn.metrics import brier_score_loss, log_loss  # standard metrics


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float).ravel()  # to float vector
    y_prob = np.asarray(y_prob).astype(float).ravel()  # to float vector
    bins = np.linspace(0.0, 1.0, n_bins + 1)  # [0..1] bins
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)  # assign bins
    total = 0.0  # accumulator
    for b in range(n_bins):  # iterate bins
        in_bin = bin_ids == b  # mask for bin b
        if not np.any(in_bin):  # skip empty bins
            continue
        p_hat = y_prob[in_bin].mean()  # avg predicted prob
        y_bar = y_true[in_bin].mean()  # empirical freq
        w = np.mean(in_bin)  # bin weight (fraction)
        total += w * abs(y_bar - p_hat)  # weighted gap
    return float(total)  # ECE scalar


def compute_core_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)  # avoid log(0)
    return {
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": ece(y_true, y_prob, n_bins=10),
    }
