# Purpose: Unit test for Elo feature builder using a tiny synthetic frame.

from __future__ import annotations  # typing future

from datetime import datetime, timezone  # timestamps

import pandas as pd  # dataframes
from esports_quant.features.elo import build_elo_features  # function under test


def test_build_elo_features_shape() -> None:
    df = pd.DataFrame(  # minimal 3-match sample
        {
            "match_id": [1, 2, 3],
            "team_a_id": [10, 10, 20],
            "team_b_id": [20, 30, 10],
            "team_a_win": [1, 0, 1],
            "start_time": [
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 2, tzinfo=timezone.utc),
                datetime(2025, 1, 3, tzinfo=timezone.utc),
            ],
            "league_name": ["L", "L", "L"],
            "bo": [1, 1, 1],
        }
    )
    out = build_elo_features(df)  # build features
    req = {  # required columns
        "match_id",
        "start_time",
        "league_name",
        "bo",
        "team_a_id",
        "team_b_id",
        "team_a_rating",
        "team_b_rating",
        "rating_diff",
        "team_a_win",
    }
    assert req.issubset(set(out.columns))  # columns present
    assert len(out) == 3  # shape preserved
