# src/esports_quant/features/build_features.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Literal

import pandas as pd


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _ensure_series(
    df: pd.DataFrame,
    col: str,
    *,
    default: float | int | str | None = None,
    dtype: Literal["Int64", "float", "object"] | None = None,
) -> pd.Series:
    """
    Return df[col] as a Series; if missing, return a default Series of the same length.
    Optionally cast to a pandas dtype (restricted to a few safe literals).
    """
    if col in df.columns:
        s = df[col]
        if not isinstance(s, pd.Series):
            s = pd.Series(s)
    else:
        s = pd.Series([default] * len(df))

    if dtype is not None:
        try:
            # Call astype with a concrete, literal dtype so Pylance is happy.
            if dtype == "object":
                s = s.astype("object")
            elif dtype == "Int64":
                s = s.astype("Int64")  # pandas' nullable integer
            elif dtype == "float":
                s = s.astype("float")
            else:
                # Shouldn't hit due to Literal type, but keep as fallback.
                s = s.astype("object")
        except Exception:
            s = s.astype("object")
    return s


def _to_numeric_series(s: pd.Series, *, coerce: bool = True) -> pd.Series:
    """
    Safe numeric conversion that always receives a Series and returns a Series.
    """
    if coerce:
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(s)


def _map_leagueid_to_tier_series(leagueid_series: pd.Series | None) -> pd.Series:
    """
    Map leagueid → coarse tier label. Placeholder: returns None for unknown leagues.
    Replace this with a real mapping if/when you have league metadata.
    """
    if leagueid_series is None:
        return pd.Series([], dtype="object")
    return leagueid_series.map(_LEAGUEID_TO_TIER).astype("object")


# Empty mapping by default; supply your own if you have it.
_LEAGUEID_TO_TIER: Dict[int, str] = {}


# --------------------------------------------------------------------------------------
# Minimal PRE-GAME (H0) feature view
# --------------------------------------------------------------------------------------
def build_pregame_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build pre-game (pre-draft) features from the matches table only.

    Expected columns in `matches` (best-effort if missing):
      - match_id, rating_diff, bo, patch, leagueid, start_time_unix, radiant_win

    Output columns:
      - match_id, rating_diff, bo, patch, league_tier, start_time, team_a_win
    """
    df = matches.copy()

    match_id = _ensure_series(df, "match_id")
    rating_diff = _to_numeric_series(_ensure_series(df, "rating_diff", default=0.0))
    bo = _to_numeric_series(_ensure_series(df, "bo", default=1)).fillna(1).astype("Int64")
    patch = _ensure_series(df, "patch", default=None, dtype="object")
    leagueid = _ensure_series(df, "leagueid")
    start_time = _to_numeric_series(_ensure_series(df, "start_time_unix", default=None)).astype("Int64")
    team_a_win = _to_numeric_series(_ensure_series(df, "radiant_win", default=0)).fillna(0).astype(int)

    league_tier = _map_leagueid_to_tier_series(leagueid)

    out = pd.DataFrame(
        {
            "match_id": match_id,
            "rating_diff": rating_diff,
            "bo": bo,
            "patch": patch,
            "league_tier": league_tier,
            "start_time": start_time,
            "team_a_win": team_a_win,
        }
    )

    out = out.dropna(subset=["match_id"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# Minimal POST-DRAFT (H1) feature view
# --------------------------------------------------------------------------------------
def build_postdraft_features(matches: pd.DataFrame, picks: pd.DataFrame, hero_wr: pd.DataFrame) -> pd.DataFrame:
    """
    Build post-draft (pre-horn) features. Uses picks and a hero winrate per patch table.

    Expected columns:
      matches: match_id, rating_diff, bo, patch, leagueid, start_time_unix, radiant_win
      picks:   match_id, is_pick (bool/int), hero_id, team (0=radiant, 1=dire)
      hero_wr: hero_id, patch, win_rate  (from scripts/build_meta.py)

    Output columns:
      - match_id, rating_diff, bo, patch, league_tier, start_time, team_a_win
      - draft_wr_sum_radiant, draft_wr_avg_radiant, draft_wr_sum_dire, draft_wr_avg_dire
    """
    # Keep only picks (exclude bans) if column exists
    p = picks.copy()
    if "is_pick" in p.columns:
        p = p[p["is_pick"] == True].copy()  # noqa: E712

    required_p = {"match_id", "team", "hero_id"}
    missing_p = required_p - set(p.columns)
    if missing_p:
        raise ValueError(f"picks is missing required columns: {sorted(missing_p)}")

    required_h = {"hero_id", "patch", "win_rate"}
    missing_h = required_h - set(hero_wr.columns)
    if missing_h:
        raise ValueError(f"hero_wr is missing required columns: {sorted(missing_h)}")

    # Attach match patch to picks so we can look up WR by (hero_id, patch)
    matches_min = matches[["match_id", "patch"]].drop_duplicates()
    picks_with_patch = p.merge(matches_min, on="match_id", how="left")

    # Join hero WR
    picks_wr = picks_with_patch.merge(
        hero_wr[["hero_id", "patch", "win_rate"]],
        on=["hero_id", "patch"],
        how="left",
    )

    # Aggregate WR per team per match
    draft_stats = (
        picks_wr.groupby(["match_id", "team"], dropna=False)["win_rate"]
        .agg(draft_wr_sum="sum", draft_wr_avg="mean")
        .reset_index()
    )

    # Pivot team (0/1) into columns for radiant/dire
    if not draft_stats.empty:
        pivot = draft_stats.pivot(index="match_id", columns="team")
        pivot.columns = [
            f"{stat}_{'radiant' if int(team) == 0 else 'dire'}" for (stat, team) in pivot.columns.to_flat_index()
        ]
        pivot = pivot.reset_index()
    else:
        pivot = pd.DataFrame(
            columns=[
                "match_id",
                "draft_wr_sum_radiant",
                "draft_wr_avg_radiant",
                "draft_wr_sum_dire",
                "draft_wr_avg_dire",
            ]
        )

    df = matches.merge(pivot, on="match_id", how="left")

    # Core columns as Series
    match_id = _ensure_series(df, "match_id")
    rating_diff = _to_numeric_series(_ensure_series(df, "rating_diff", default=0.0))
    bo = _to_numeric_series(_ensure_series(df, "bo", default=1)).fillna(1).astype("Int64")
    patch = _ensure_series(df, "patch", default=None, dtype="object")
    leagueid = _ensure_series(df, "leagueid")
    start_time = _to_numeric_series(_ensure_series(df, "start_time_unix", default=None)).astype("Int64")
    team_a_win = _to_numeric_series(_ensure_series(df, "radiant_win", default=0)).fillna(0).astype(int)

    league_tier = _map_leagueid_to_tier_series(leagueid)

    # Draft WR columns (ensure Series even if missing)
    d_sum_r = _to_numeric_series(_ensure_series(df, "draft_wr_sum_radiant", default=None))
    d_avg_r = _to_numeric_series(_ensure_series(df, "draft_wr_avg_radiant", default=None))
    d_sum_d = _to_numeric_series(_ensure_series(df, "draft_wr_sum_dire", default=None))
    d_avg_d = _to_numeric_series(_ensure_series(df, "draft_wr_avg_dire", default=None))

    out = pd.DataFrame(
        {
            "match_id": match_id,
            "rating_diff": rating_diff,
            "bo": bo,
            "patch": patch,
            "league_tier": league_tier,
            "draft_wr_sum_radiant": d_sum_r,
            "draft_wr_avg_radiant": d_avg_r,
            "draft_wr_sum_dire": d_sum_d,
            "draft_wr_avg_dire": d_avg_d,
            "start_time": start_time,
            "team_a_win": team_a_win,
        }
    )

    out = out.dropna(subset=["match_id"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# Back-compat: combined builder proxies to post-draft when WR is present; else pre-game
# --------------------------------------------------------------------------------------
def build_match_team_features(
    matches: pd.DataFrame, picks: pd.DataFrame, players: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Minimal placeholder that proxies to post-draft features for backward compatibility.
    If hero WR parquet is unavailable, falls back to pre-game features.
    """
    default_wr = Path("data/processed/hero_patch_winrates.parquet")
    if default_wr.exists():
        hero_wr = pd.read_parquet(default_wr)
        return build_postdraft_features(matches, picks, hero_wr)
    else:
        return build_pregame_features(matches)
