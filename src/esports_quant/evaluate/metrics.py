from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class MetricBundle:
    log_loss: float
    brier: float
    ece: float


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    ECE = sum_k (n_k / N) * |mean(pred_k) - frac_pos_k|, using sklearn's binning.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        inds = np.clip(np.digitize(y_prob, edges) - 1, 0, n_bins - 1)
    else:
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        inds = np.clip(np.digitize(y_prob, edges, right=True) - 1, 0, n_bins - 1)

    counts = np.bincount(inds, minlength=n_bins).astype(float)
    weights = counts / max(1, counts.sum())

    ece = 0.0
    cursor = 0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        gap = abs(mean_pred[cursor] - frac_pos[cursor])
        ece += weights[b] * gap
        cursor += 1
    return float(ece)


def bundle_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> MetricBundle:
    return MetricBundle(
        log_loss=log_loss(y_true, np.clip(y_prob, 1e-12, 1 - 1e-12)),
        brier=brier_score_loss(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy),
    )
