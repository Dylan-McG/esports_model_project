from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Config / paths
# --------------------------------------------------------------------------------------

RAW_DIR = Path("data/raw")
RAW_PROMATCHES_DIR = RAW_DIR / "proMatches"
RAW_MATCHDETAILS_DIR = RAW_DIR / "matchDetails"

PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

OUT_MATCHES = PROC_DIR / "matches.parquet"
OUT_PICKS = PROC_DIR / "match_picks.parquet"
OUT_PLAYERS = PROC_DIR / "match_players.parquet"

# New: event/longform outputs
OUT_OBJ = PROC_DIR / "events_objectives.parquet"
OUT_KILLS = PROC_DIR / "events_kills.parquet"
OUT_ITEMS = PROC_DIR / "events_items.parquet"
OUT_ABIL = PROC_DIR / "events_abilities.parquet"
OUT_WARDS = PROC_DIR / "events_wards.parquet"
OUT_TF = PROC_DIR / "teamfights.parquet"
OUT_PTS = PROC_DIR / "players_timeseries.parquet"

# --------------------------------------------------------------------------------------
# OpenDota client (simple)
# --------------------------------------------------------------------------------------

import requests

BASE_URL = "https://api.opendota.com/api"
API_KEY = os.getenv("OPENDOTA_API_KEY", "").strip()


def _request(path: str, params: Optional[Dict[str, Any]] = None, sleep_s: float = 0.08) -> requests.Response:
    """GET request with api_key if present."""
    url = f"{BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    params = dict(params or {})
    if API_KEY:
        # Support both query parameter and Bearer header; query is simplest here.
        params["api_key"] = API_KEY
        headers = {"Authorization": f"Bearer {API_KEY}"}
    else:
        headers = {}
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    # gentle pacing
    if sleep_s > 0:
        time.sleep(sleep_s)
    return resp


# --------------------------------------------------------------------------------------
# Pro matches list + details cache
# --------------------------------------------------------------------------------------


def fetch_promatches_page(less_than_match_id: Optional[int] = None) -> List[Dict[str, Any]]:
    params = {}
    if less_than_match_id is not None:
        params["less_than_match_id"] = int(less_than_match_id)
    r = _request("/proMatches", params=params)
    r.raise_for_status()
    return r.json()


def fetch_match_detail(match_id: int) -> Dict[str, Any]:
    r = _request(f"/matches/{match_id}")
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------------------
# Patch constants (array or dict in API; normalize to a list of dicts)
# --------------------------------------------------------------------------------------


def fetch_patch_constants() -> List[Dict[str, Any]]:
    r = _request("/constants/patch")
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        # Older shape (map name -> meta); convert to list with 'name' field
        out = []
        for name, meta in data.items():
            m = dict(meta or {})
            m["name"] = name
            out.append(m)
        return out
    elif isinstance(data, list):
        return data
    else:
        return []


def build_patch_lookup(patch_const: List[Dict[str, Any]]):
    """
    Return a function: unix_timestamp -> patch name (e.g., '7.39')
    Patch constants contain 'name', 'date' (ISO) and 'timestamp' (unix seconds).
    We'll map each start_time to the latest patch whose timestamp <= start_time.
    """
    df = pd.DataFrame(patch_const)
    # Some builds have 'timestamp' and 'name'; ensure correct types.
    if "timestamp" not in df.columns:
        return lambda ts: None
    df = df[["name", "timestamp"]].dropna()
    df = df.sort_values("timestamp")
    stamps = df["timestamp"].to_numpy()
    names = df["name"].to_numpy()

    def _lookup(ts: Optional[int]) -> Optional[str]:
        if ts is None or np.isnan(ts):
            return None
        ts = int(ts)
        idx = np.searchsorted(stamps, ts, side="right") - 1
        if idx < 0:
            return None
        return str(names[idx])

    return _lookup


# --------------------------------------------------------------------------------------
# Helpers to read raw cache
# --------------------------------------------------------------------------------------


def load_cached_promatches_ids(limit_ids: Optional[int] = None) -> List[int]:
    """
    Read any saved proMatches pages then (if still under limit) continue fetching pages.
    Cache as page_{n}.json. Return match_id list (descending time).
    """
    RAW_PROMATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing pages
    pages: List[List[Dict[str, Any]]] = []
    for p in sorted(RAW_PROMATCHES_DIR.glob("page_*.json")):
        try:
            pages.append(json.loads(p.read_text()))
        except Exception:
            continue

    ids: List[int] = []
    for pg in pages:
        ids.extend([int(x["match_id"]) for x in pg if "match_id" in x])

    # If limit met by cache, stop
    if limit_ids is not None and len(ids) >= limit_ids:
        return ids[:limit_ids]

    # Otherwise, continue paging from last min(match_id)
    last_lt = min(ids) if ids else None
    page_i = len(pages)
    # Keep fetching until limit reached (or up to a safety max)
    while True:
        data = fetch_promatches_page(less_than_match_id=last_lt)
        if not data:
            break
        (RAW_PROMATCHES_DIR / f"page_{page_i}.json").write_text(json.dumps(data))
        page_i += 1
        new_ids = [int(x["match_id"]) for x in data if "match_id" in x]
        if not new_ids:
            break
        ids.extend(new_ids)
        last_lt = min(new_ids)
        if limit_ids is not None and len(ids) >= limit_ids:
            break

    return ids[:limit_ids] if limit_ids else ids


def cache_match_details_for_ids(ids: Iterable[int]) -> Tuple[int, int, List[int], List[int]]:
    """
    For each match_id: if not in cache, fetch and save.
    Returns: (cached_count, skipped_count, sample_404, sample_invalid)
    """
    RAW_MATCHDETAILS_DIR.mkdir(parents=True, exist_ok=True)
    cached = 0
    skipped = 0
    sample_404: List[int] = []
    sample_invalid: List[int] = []

    for i, mid in enumerate(ids, 1):
        f = RAW_MATCHDETAILS_DIR / f"{mid}.json"
        if f.exists():
            skipped += 1
        else:
            try:
                data = fetch_match_detail(int(mid))
                # sanity: must contain players and radiant_gold_adv at minimum for our pipeline
                if not isinstance(data, dict) or "players" not in data:
                    sample_invalid.append(int(mid))
                    continue
                f.write_text(json.dumps(data))
                cached += 1
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    sample_404.append(int(mid))
                # else just skip silently; could add logging
            except Exception:
                sample_invalid.append(int(mid))

        if i % 1000 == 0:
            print(f"[cache] progress: {i} / {len(list(ids))} (cached={cached}, skipped={skipped})")

    return cached, skipped, sample_404[:5], sample_invalid[:5]


# --------------------------------------------------------------------------------------
# Tidy builders
# --------------------------------------------------------------------------------------


def _gold_slope(arr: Optional[List[float]], t0: int, t1: int) -> Optional[float]:
    if not arr:
        return None
    if t0 < 0:
        t0 = 0
    if t1 <= t0:
        return None
    if t1 >= len(arr):
        t1 = len(arr) - 1
    if t1 <= t0:
        return None
    y0 = arr[t0]
    y1 = arr[t1]
    if y0 is None or y1 is None:
        return None
    try:
        return (y1 - y0) / (t1 - t0)
    except Exception:
        return None


def _safe(lst: Optional[List[Any]], idx: int) -> Optional[Any]:
    if not isinstance(lst, list):
        return None
    if idx < 0 or idx >= len(lst):
        return None
    return lst[idx]


def _extract_picks_bans(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for pb in match.get("picks_bans", []) or []:
        row = {
            "match_id": match.get("match_id"),
            "is_pick": pb.get("is_pick"),
            "hero_id": pb.get("hero_id"),
            "team": pb.get("team"),
            "order": pb.get("order"),
        }
        out.append(row)
    return out


def _extract_players(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for pl in match.get("players", []) or []:
        row = {
            "match_id": match.get("match_id"),
            "account_id": pl.get("account_id"),
            "player_slot": pl.get("player_slot"),
            "hero_id": pl.get("hero_id"),
            "kills": pl.get("kills"),
            "deaths": pl.get("deaths"),
            "assists": pl.get("assists"),
            "gold_per_min": pl.get("gold_per_min"),
            "xp_per_min": pl.get("xp_per_min"),
            "tower_damage": pl.get("tower_damage"),
            "stuns": pl.get("stuns"),
            "actions_per_min": pl.get("actions_per_min"),
            "lane": pl.get("lane"),
            "lane_role": pl.get("lane_role"),
            "is_roaming": pl.get("is_roaming"),
            "lane_efficiency_pct": pl.get("lane_efficiency_pct"),
            "win": pl.get("win"),
            "isRadiant": pl.get("isRadiant"),
            "rank_tier": pl.get("rank_tier"),
        }
        rows.append(row)
    return rows


def _extract_objectives(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract objective/timeline events from a match, normalizing types so we can write Parquet safely.
    OpenDota 'objectives' is an array of dicts with heterogeneous keys; importantly:
      - 'type' is a string (e.g., 'building_kill', 'CHAT_MESSAGE_ROSHAN_KILL', etc.)
      - 'key' may be an int OR a string (e.g., building name); we coerce to STRING.
      - 'time' is seconds from game start (can be int); we coerce to int when possible.
      - 'player_slot' / 'slot' may appear; we try to coerce to int, else NA.
      - 'team' often 0 (Radiant) or 1 (Dire); we coerce to int when possible, else NA.

    Output schema (stable, Parquet-friendly):
      match_id:int64
      time_s:int32 (nullable)
      type:string
      key:string
      team:Int8 (nullable)
      player_slot:Int16 (nullable)
      slot:Int16 (nullable)
    """
    from math import isnan

    rows: List[Dict[str, Any]] = []
    objs = match.get("objectives")
    if not isinstance(objs, list):
        return rows

    mid = match.get("match_id")

    def _to_int_or_na(v) -> Optional[int]:
        try:
            if v is None:
                return None
            # strings like "12" or actual ints
            iv = int(v)
            return iv
        except Exception:
            return None

    for ev in objs:
        if not isinstance(ev, dict):
            continue

        # time
        t = ev.get("time")
        t_int: Optional[int] = None
        try:
            # some payloads are float-y; clamp to int seconds
            t_int = int(t) if t is not None and not (isinstance(t, float) and isnan(t)) else None
        except Exception:
            t_int = None

        # normalize type/key -> strings
        ev_type = ev.get("type")
        ev_key = ev.get("key")

        # force string type for both
        type_str = "" if ev_type is None else str(ev_type)
        key_str = "" if ev_key is None else str(ev_key)

        row = {
            "match_id": mid,
            "time_s": t_int,
            "type": type_str,
            "key": key_str,
            "team": _to_int_or_na(ev.get("team")),
            "player_slot": _to_int_or_na(ev.get("player_slot")),
            "slot": _to_int_or_na(ev.get("slot")),
        }
        rows.append(row)

    return rows


def _extract_kills(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pl in match.get("players", []) or []:
        killer_acc = pl.get("account_id")
        killer_slot = pl.get("player_slot")
        killer_hero = pl.get("hero_id")
        for k in pl.get("kills_log", []) or []:
            # kills_log items often have {'time': int, 'key': 'npc_dota_hero_*'}
            rows.append(
                {
                    "match_id": match.get("match_id"),
                    "time_s": k.get("time"),
                    "killer_account_id": killer_acc,
                    "killer_player_slot": killer_slot,
                    "killer_hero_id": killer_hero,
                    "victim_key": k.get("key"),  # hero string; victim_id not always present
                }
            )
    return rows


def _extract_items(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pl in match.get("players", []) or []:
        acc = pl.get("account_id")
        hero = pl.get("hero_id")
        slot = pl.get("player_slot")
        for it in pl.get("purchase_log", []) or []:
            rows.append(
                {
                    "match_id": match.get("match_id"),
                    "account_id": acc,
                    "player_slot": slot,
                    "hero_id": hero,
                    "time_s": it.get("time"),
                    "item": it.get("key"),
                    "charges": it.get("charges"),
                }
            )
    return rows


def _extract_abilities(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prefer 'ability_upgrades' (with time/level) if present; otherwise fall back to 'ability_upgrades_arr'
    (only order, no timestamps).
    """
    rows: List[Dict[str, Any]] = []
    for pl in match.get("players", []) or []:
        acc = pl.get("account_id")
        hero = pl.get("hero_id")
        slot = pl.get("player_slot")
        # Rich structure sometimes available:
        ab_up = pl.get("ability_upgrades")
        if isinstance(ab_up, list) and ab_up:
            for a in ab_up:
                rows.append(
                    {
                        "match_id": match.get("match_id"),
                        "account_id": acc,
                        "player_slot": slot,
                        "hero_id": hero,
                        "ability_id": a.get("ability"),
                        "level": a.get("level"),
                        "time_s": a.get("time"),
                    }
                )
            continue
        # Fallback: just the order
        arr = pl.get("ability_upgrades_arr") or []
        for lvl, ability_id in enumerate(arr, start=1):
            rows.append(
                {
                    "match_id": match.get("match_id"),
                    "account_id": acc,
                    "player_slot": slot,
                    "hero_id": hero,
                    "ability_id": ability_id,
                    "level": lvl,
                    "time_s": None,
                }
            )
    return rows


def _extract_wards(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pl in match.get("players", []) or []:
        acc = pl.get("account_id")
        hero = pl.get("hero_id")
        slot = pl.get("player_slot")

        def _rows(log: Optional[List[Dict[str, Any]]], ward_type: str):
            if not isinstance(log, list):
                return
            for w in log:
                rows.append(
                    {
                        "match_id": match.get("match_id"),
                        "account_id": acc,
                        "player_slot": slot,
                        "hero_id": hero,
                        "type": ward_type,  # 'obs' or 'sen'
                        "time_s": w.get("time"),
                        "x": w.get("x"),
                        "y": w.get("y"),
                        "z": w.get("z"),
                        "attackername": w.get("attackername"),
                    }
                )

        _rows(pl.get("obs_log"), "obs")
        _rows(pl.get("sen_log"), "sen")
    return rows


def _extract_teamfights(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tfs = match.get("teamfights")
    if not isinstance(tfs, list):
        return rows

    for tf_id, tf in enumerate(tfs):
        start = tf.get("start")
        end = tf.get("end")
        deaths = tf.get("deaths")
        players = tf.get("players")

        # Default aggregates
        rad_dmg = 0
        dire_dmg = 0
        rad_deaths = None
        dire_deaths = None

        # Handle deaths: list of player_slots OR single int
        total_deaths: Optional[int] = None
        if isinstance(deaths, list):
            total_deaths = len(deaths)
            r, d = 0, 0
            for val in deaths:
                try:
                    ps = int(val)
                    if ps < 128:
                        r += 1
                    else:
                        d += 1
                except Exception:
                    pass
            rad_deaths, dire_deaths = r, d
        elif isinstance(deaths, int):
            total_deaths = deaths

        # Handle players: list OR dict keyed by slot
        if isinstance(players, list):
            for i, p in enumerate(players):
                dmg = p.get("damage", 0) or 0
                ps = p.get("player_slot")
                if ps is None:
                    # Heuristic: indexes 0..4 radiant, 5..9 dire
                    if i < 5:
                        rad_dmg += dmg
                    else:
                        dire_dmg += dmg
                else:
                    try:
                        if int(ps) < 128:
                            rad_dmg += dmg
                        else:
                            dire_dmg += dmg
                    except Exception:
                        pass
        elif isinstance(players, dict):
            for k, p in players.items():
                dmg = (p or {}).get("damage", 0) or 0
                try:
                    ps = int(k) if str(k).isdigit() else (p.get("player_slot"))
                    if ps is not None and int(ps) < 128:
                        rad_dmg += dmg
                    else:
                        dire_dmg += dmg
                except Exception:
                    pass

        rows.append(
            {
                "match_id": match.get("match_id"),
                "tf_id": tf_id,
                "start_s": start,
                "end_s": end,
                "rad_dmg": rad_dmg,
                "dire_dmg": dire_dmg,
                "rad_deaths": rad_deaths,
                "dire_deaths": dire_deaths,
                "total_deaths": total_deaths,
            }
        )
    return rows


def _extract_players_timeseries(match: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pl in match.get("players", []) or []:
        acc = pl.get("account_id")
        hero = pl.get("hero_id")
        slot = pl.get("player_slot")
        gold = pl.get("gold_t") or []
        xp = pl.get("xp_t") or []
        lh = pl.get("lh_t") or []
        dn = pl.get("dn_t") or []
        maxlen = max(len(gold), len(xp), len(lh), len(dn), 0)
        for minute in range(maxlen):
            rows.append(
                {
                    "match_id": match.get("match_id"),
                    "account_id": acc,
                    "player_slot": slot,
                    "hero_id": hero,
                    "minute": minute,
                    "gold": _safe(gold, minute),
                    "xp": _safe(xp, minute),
                    "last_hits": _safe(lh, minute),
                    "denies": _safe(dn, minute),
                }
            )
    return rows


def _match_row(match: Dict[str, Any], to_patch) -> Dict[str, Any]:
    rga = match.get("radiant_gold_adv") or []
    # Simple slopes & snapshots
    gold_adv_10 = _safe(rga, 10)
    gold_adv_20 = _safe(rga, 20)
    gold_adv_30 = _safe(rga, 30)
    slope_0_10 = _gold_slope(rga, 0, 10)
    slope_10_25 = _gold_slope(rga, 10, 25)
    slope_25_end = _gold_slope(rga, 25, len(rga) - 1 if len(rga) > 25 else 25)

    # Best-of from series_type (1=bo1, 2=bo3, 3=bo5 typically)
    series_type = match.get("series_type")
    bo = {1: 1, 2: 3, 3: 5}.get(series_type, 1)

    # Rating diff placeholder (0 if not computed elsewhere yet)
    rating_diff = 0.0

    start_time_unix = match.get("start_time")
    patch = to_patch(start_time_unix) if to_patch else None

    return {
        "match_id": match.get("match_id"),
        "duration": match.get("duration"),
        "start_time_unix": start_time_unix,
        "start_time": pd.to_datetime(start_time_unix, unit="s", utc=True) if start_time_unix else pd.NaT,
        "radiant_team_id": match.get("radiant_team_id"),
        "radiant_name": match.get("radiant_name"),
        "dire_team_id": match.get("dire_team_id"),
        "dire_name": match.get("dire_name"),
        "leagueid": match.get("leagueid"),
        "league_name": match.get("league", {}).get("name")
        if isinstance(match.get("league"), dict)
        else match.get("league_name"),
        "series_id": match.get("series_id"),
        "series_type": series_type,
        "bo": bo,
        "radiant_score": match.get("radiant_score"),
        "dire_score": match.get("dire_score"),
        "radiant_win": match.get("radiant_win"),
        "patch": patch,
        "gold_adv_10": gold_adv_10,
        "gold_adv_20": gold_adv_20,
        "gold_adv_30": gold_adv_30,
        "gold_slope_0_10": slope_0_10,
        "gold_slope_10_25": slope_10_25,
        "gold_slope_25_end": slope_25_end,
        "rating_diff": rating_diff,
        "comeback_flag": bool(
            (gold_adv_10 and gold_adv_20)
            and (
                (gold_adv_10 < 0 and gold_adv_30 and gold_adv_30 > 0)
                or (gold_adv_10 > 0 and gold_adv_30 and gold_adv_30 < 0)
            )
        ),
    }


def build_tidy_from_cache(
    cfg: "BuildConfig",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Returns (matches_df, picks_df, players_df_or_none, extra_tables_dict)
    """
    patch_const = fetch_patch_constants()
    to_patch = build_patch_lookup(patch_const)

    # Collect rows
    matches_rows: List[Dict[str, Any]] = []
    picks_rows: List[Dict[str, Any]] = []
    players_rows: List[Dict[str, Any]] = []

    obj_rows: List[Dict[str, Any]] = []
    kill_rows: List[Dict[str, Any]] = []
    item_rows: List[Dict[str, Any]] = []
    abil_rows: List[Dict[str, Any]] = []
    ward_rows: List[Dict[str, Any]] = []
    tf_rows: List[Dict[str, Any]] = []
    pts_rows: List[Dict[str, Any]] = []

    # Iterate cached details
    for f in RAW_MATCHDETAILS_DIR.glob("*.json"):
        try:
            m = json.loads(f.read_text())
        except Exception:
            continue

        # core match row
        matches_rows.append(_match_row(m, to_patch))
        # draft
        picks_rows.extend(_extract_picks_bans(m))
        # optional players
        if cfg.cache_players:
            players_rows.extend(_extract_players(m))
        # extras
        obj_rows.extend(_extract_objectives(m))
        kill_rows.extend(_extract_kills(m))
        item_rows.extend(_extract_items(m))
        abil_rows.extend(_extract_abilities(m))
        ward_rows.extend(_extract_wards(m))
        tf_rows.extend(_extract_teamfights(m))
        pts_rows.extend(_extract_players_timeseries(m))

    # DataFrames
    matches_df = pd.DataFrame(matches_rows)
    picks_df = pd.DataFrame(picks_rows)
    players_df = pd.DataFrame(players_rows) if cfg.cache_players else None

    tables = {
        "events_objectives": pd.DataFrame(obj_rows),
        "events_kills": pd.DataFrame(kill_rows),
        "events_items": pd.DataFrame(item_rows),
        "events_abilities": pd.DataFrame(abil_rows),
        "events_wards": pd.DataFrame(ward_rows),
        "teamfights": pd.DataFrame(tf_rows),
        "players_timeseries": pd.DataFrame(pts_rows),
    }
    return matches_df, picks_df, players_df, tables


# --------------------------------------------------------------------------------------
# Main build flow
# --------------------------------------------------------------------------------------


@dataclass
class BuildConfig:
    limit_ids: Optional[int] = None
    cache_players: bool = False


def run_full_build(cfg: BuildConfig) -> Dict[str, Any]:
    # 1) IDs
    ids = load_cached_promatches_ids(limit_ids=cfg.limit_ids)

    # 2) Details cache
    cached, skipped, sample_404, sample_invalid = cache_match_details_for_ids(ids)

    # 3) Tidy
    matches, picks, players_df, tables = build_tidy_from_cache(cfg)

    # 4) Write
    matches.to_parquet(OUT_MATCHES, index=False)
    picks.to_parquet(OUT_PICKS, index=False)
    players_rows = 0
    players_out = None
    if cfg.cache_players and players_df is not None and not players_df.empty:
        players_df.to_parquet(OUT_PLAYERS, index=False)
        players_rows = len(players_df)
        players_out = str(OUT_PLAYERS)

    # Extra tables
    if not tables["events_objectives"].empty:
        tables["events_objectives"].to_parquet(OUT_OBJ, index=False)
    if not tables["events_kills"].empty:
        tables["events_kills"].to_parquet(OUT_KILLS, index=False)
    if not tables["events_items"].empty:
        tables["events_items"].to_parquet(OUT_ITEMS, index=False)
    if not tables["events_abilities"].empty:
        tables["events_abilities"].to_parquet(OUT_ABIL, index=False)
    if not tables["events_wards"].empty:
        tables["events_wards"].to_parquet(OUT_WARDS, index=False)
    if not tables["teamfights"].empty:
        tables["teamfights"].to_parquet(OUT_TF, index=False)
    if not tables["players_timeseries"].empty:
        tables["players_timeseries"].to_parquet(OUT_PTS, index=False)

    report = {
        "promatches_ids": len(ids),
        "details_cached": cached + skipped,
        "skipped_404_count": len(sample_404),
        "skipped_invalid_count": len(sample_invalid),
        "sample_skipped_404": sample_404,
        "sample_skipped_invalid": sample_invalid,
        "missing_details_sample": [],  # could compute by comparing ids vs cached file names
        "matches_rows": len(matches),
        "picks_rows": len(picks),
        "players_rows": players_rows,
        "matches_out": str(OUT_MATCHES),
        "picks_out": str(OUT_PICKS),
        "players_out": players_out,
        "events_objectives_rows": len(tables["events_objectives"]),
        "events_kills_rows": len(tables["events_kills"]),
        "events_items_rows": len(tables["events_items"]),
        "events_abilities_rows": len(tables["events_abilities"]),
        "events_wards_rows": len(tables["events_wards"]),
        "teamfights_rows": len(tables["teamfights"]),
        "players_timeseries_rows": len(tables["players_timeseries"]),
        "events_objectives_out": str(OUT_OBJ),
        "events_kills_out": str(OUT_KILLS),
        "events_items_out": str(OUT_ITEMS),
        "events_abilities_out": str(OUT_ABIL),
        "events_wards_out": str(OUT_WARDS),
        "teamfights_out": str(OUT_TF),
        "players_timeseries_out": str(OUT_PTS),
    }
    print(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OpenDota ingest (IDs → details cache → tidy + events parquet)")
    ap.add_argument("--limit-ids", type=int, default=None, help="Limit number of pro match IDs to fetch/cache")
    ap.add_argument("--players", action="store_true", help="Also write match_players.parquet")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rep = run_full_build(BuildConfig(limit_ids=args.limit_ids, cache_players=args.players))
