# ruff: noqa: D401
"""
High-level OpenDota ingest pipeline.

Responsibilities
----------------
- Use the low-level data client (esports_quant.data.opendota) to pull pro match data.
- Engineer baseline features (Elo pre-match rating diff).
- Output a processed Parquet dataset suitable for training/evaluation.
- Keep ingestion idempotent and reproducible.

Notes
-----
- Elo is computed sequentially over time; re-running on a larger dataset will
  slightly change earlier ratings if new older matches are included.
- For large historical pulls, consider caching the processed parquet and only
  appending new matches (with Elo recompute if needed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from esports_quant.data.opendota import fetch_pro_matches

# ---------------------------
# Elo helpers
# ---------------------------


def _elo_expected(r_a: float, r_b: float) -> float:
    """Win probability of player/team A under Elo."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _apply_inline_elo(df: pd.DataFrame, k: float = 20.0, base: float = 1500.0) -> pd.Series:
    """
    Compute a simple Elo per team as we walk forward in time and emit
    rating_diff = Elo(team_a) - Elo(team_b) at each row.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `team_a_id`, `team_b_id`, `team_a_win`, sorted by start_time.
    k : float
        Elo K-factor (sensitivity).
    base : float
        Starting rating for unseen teams.

    Returns
    -------
    pd.Series
        rating_diff values aligned to df.index
    """
    ratings: Dict[int, float] = {}
    diffs: list[float] = []

    for row in df.itertuples(index=False):
        a = int(row.team_a_id)
        b = int(row.team_b_id)
        y = int(row.team_a_win)

        r_a = ratings.get(a, base)
        r_b = ratings.get(b, base)

        # Emit pre-game diff
        diffs.append(r_a - r_b)

        # Update after match
        p_a = _elo_expected(r_a, r_b)
        s_a = 1.0 if y == 1 else 0.0
        s_b = 1.0 - s_a

        ratings[a] = r_a + k * (s_a - p_a)
        ratings[b] = r_b + k * (s_b - (1.0 - p_a))

    return pd.Series(diffs, index=df.index, name="rating_diff")


# ---------------------------
# Main ingest
# ---------------------------


def ingest_pro_matches(
    limit: int = 50000,
    processed_path: Path = Path("data/processed/matches.parquet"),
) -> Path:
    """
    Fetch, normalize, engineer Elo, and write Parquet.

    Parameters
    ----------
    limit : int
        Number of most-recent pro matches to fetch.
    processed_path : Path
        Output Parquet path for processed dataset.

    Returns
    -------
    Path
        The written Parquet path.
    """
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Fetch + normalize from low-level client
    df = fetch_pro_matches(limit=limit, page_size=100)

    if df.empty:
        raise RuntimeError("No matches returned from OpenDota.")

    # 2) Ensure sorted by start_time for Elo
    df = df.sort_values("start_time").reset_index(drop=True)

    # 3) Compute Elo diff
    df["rating_diff"] = _apply_inline_elo(df)

    # 4) Select features for baseline model
    feats = [
        "match_id",
        "start_time",
        "team_a_id",
        "team_b_id",
        "bo",
        "rating_diff",
        "team_a_win",
    ]
    out_df = df[feats]

    # 5) Write Parquet
    out_df.to_parquet(processed_path, index=False)

    return processed_path


# ---------------------------
# CLI entry point (optional)
# ---------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest OpenDota pro matches to Parquet.")
    parser.add_argument("--limit", type=int, default=50000, help="Number of matches to fetch.")
    parser.add_argument("--out", type=str, default="data/processed/matches.parquet", help="Output Parquet file.")
    args = parser.parse_args()

    path = ingest_pro_matches(limit=args.limit, processed_path=Path(args.out))
    print(f"Wrote processed dataset to: {path}")
