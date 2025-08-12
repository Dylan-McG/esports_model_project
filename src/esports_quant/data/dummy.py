# Purpose: Deterministic dummy dataset for Day-1 end-to-end pipeline.

from __future__ import annotations  # typing future behavior

from pathlib import Path  # file paths

import numpy as np  # RNG and math
import pandas as pd  # tabular data

from ..utils.io import ensure_dir  # dir helper


def make_dummy_matches(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)  # reproducible RNG
    team_a_rating = rng.normal(1500, 150, size=n)  # Team A Elo-like rating
    team_b_rating = rng.normal(1500, 150, size=n)  # Team B Elo-like rating
    bo = rng.choice([1, 3], size=n, p=[0.6, 0.4])  # Bo1 or Bo3
    rating_diff = team_a_rating - team_b_rating  # simple signal
    logit = 0.004 * rating_diff + 0.15 * (bo == 3)  # true model (hidden)
    p = 1.0 / (1.0 + np.exp(-logit))  # logistic to probability
    team_a_win = rng.binomial(1, p, size=n)  # outcome draw
    df = pd.DataFrame(
        {
            "match_id": np.arange(1, n + 1, dtype=int),
            "team_a_rating": team_a_rating,
            "team_b_rating": team_b_rating,
            "bo": bo,
            "rating_diff": rating_diff,
            "team_a_win": team_a_win,
        }
    )
    return df  # return DataFrame


def ingest_to_parquet(out_dir: str | Path = "data/processed", seed: int = 42) -> Path:
    out_path = ensure_dir(out_dir) / "matches.parquet"  # target path
    df = make_dummy_matches(seed=seed)  # generate data
    df.to_parquet(out_path, index=False)  # write Parquet file
    return out_path  # return written path
