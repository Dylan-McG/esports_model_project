# Purpose: Compute simple Elo for teams over time and emit per-match features.

from __future__ import annotations  # future typing

from typing import Dict  # type hint for Elo table

import pandas as pd  # tabular

INIT_ELO = 1500.0  # starting rating for unseen team
K = 20.0  # update rate per match
HOME_ADV = 0.0  # no home advantage in esports
SCALE = 400.0  # logistic scaling (classic Elo)


def _expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b + HOME_ADV) / SCALE))  # classic Elo expected score


def build_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("start_time").reset_index(drop=True)  # time order
    df["team_a_rating"] = 0.0  # pre-match rating A
    df["team_b_rating"] = 0.0  # pre-match rating B

    elo: Dict[int, float] = {}  # team_id -> rating

    for i, row in df.iterrows():  # sequentially update
        a = int(row["team_a_id"])
        b = int(row["team_b_id"])
        y = float(row["team_a_win"])

        r_a = elo.get(a, INIT_ELO)  # current ratings
        r_b = elo.get(b, INIT_ELO)

        df.at[i, "team_a_rating"] = r_a  # store features
        df.at[i, "team_b_rating"] = r_b

        e_a = _expected_score(r_a, r_b)  # expectation
        r_a_new = r_a + K * (y - e_a)  # update A
        r_b_new = r_b + K * ((1.0 - y) - (1.0 - e_a))  # update B

        elo[a] = r_a_new  # persist
        elo[b] = r_b_new

    df["rating_diff"] = df["team_a_rating"] - df["team_b_rating"]  # derived

    cols = [
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
    ]
    return df[cols].copy()  # minimal output schema
