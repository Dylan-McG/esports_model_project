from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_dummy_matches(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(2025)
    start = np.datetime64("2025-07-01T00:00:00Z")
    times = start + rng.integers(0, 60 * 60 * 24 * 30, size=n).astype("timedelta64[s]")

    # synthetic elo diffs and best-of
    rating_diff = rng.normal(0, 50, size=n)
    bo = rng.choice([1, 3, 5], p=[0.2, 0.7, 0.1], size=n)

    # map to probability and outcome
    p = 1 / (1 + np.exp(-rating_diff / 25))  # squashed diff
    y = rng.binomial(1, p)

    df = pd.DataFrame(
        {
            "match_id": np.arange(1, n + 1, dtype=np.int64),
            "start_time": pd.to_datetime(times),
            "duration": rng.integers(1200, 3600, size=n, dtype=np.int32),
            "league_name": "DUMMY",
            "bo": bo.astype("int16"),
            "rating_diff": rating_diff.astype("float32"),
            "team_a_win": y.astype("int8"),
        }
    )
    return df.sort_values("start_time").reset_index(drop=True)


def write_dummy_processed(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matches.parquet"
    df = make_dummy_matches(1000)
    df.to_parquet(out_path, index=False)
    return out_path


# Backwards-compat alias if older code imports this name:
def generate_dummy_matches(out_dir: str | Path) -> Path:  # pragma: no cover
    return write_dummy_processed(out_dir)
