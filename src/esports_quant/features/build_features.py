from __future__ import annotations

from typing import Any

import pandas as pd


def build_match_team_features(
    matches: pd.DataFrame,
    picks: pd.DataFrame,
    players: pd.DataFrame | None = None,
    **event_tables: dict[str, Any],
) -> pd.DataFrame:
    """
    Minimal scaffold: return one row per match with a few ready-to-train features.
    - abs_rating_diff (if rating_diff exists)
    - momentum features (already in matches)
    - basic draft summaries (per side counts) if picks has 'team'/'is_pick'
    """
    feats = matches.copy()

    # Example: absolute Elo diff
    if "rating_diff" in feats.columns:
        feats["abs_rating_diff"] = feats["rating_diff"].abs()

    # Draft counts by side (requires picks: is_pick, team [0=radiant,1=dire])
    need_cols = {"match_id", "is_pick", "team"}
    if need_cols.issubset(set(picks.columns)):
        p = picks.loc[picks["is_pick"] == True, ["match_id", "team"]]  # noqa: E712
        draft_counts = p.value_counts(["match_id", "team"]).rename("pick_count").reset_index()
        wide = draft_counts.pivot(index="match_id", columns="team", values="pick_count").fillna(0.0)
        wide = wide.rename(columns={0: "radiant_picks", 1: "dire_picks"}).reset_index()
        feats = feats.merge(wide, on="match_id", how="left")
    else:
        feats["radiant_picks"] = feats.get("radiant_picks", pd.Series(index=feats.index, dtype="float64"))
        feats["dire_picks"] = feats.get("dire_picks", pd.Series(index=feats.index, dtype="float64"))

    # Ensure consistent column order (light touch)
    preferred = [
        "match_id",
        "start_time",
        "patch",
        "rating_diff",
        "abs_rating_diff",
        "gold_adv_10",
        "gold_adv_20",
        "gold_adv_30",
        "gold_slope_0_10",
        "gold_slope_10_25",
        "gold_slope_25_end",
        "radiant_picks",
        "dire_picks",
        "radiant_win",
    ]
    cols = [c for c in preferred if c in feats.columns] + [c for c in feats.columns if c not in preferred]
    return feats[cols]
