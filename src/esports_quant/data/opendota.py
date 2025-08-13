# ruff: noqa: D401
"""
Low-level OpenDota client utilities.

- Robust HTTP session with retries/backoff and polite pacing.
- Premium API key injection (via esports_quant.config).
- Paginated fetching of /proMatches.
- Minimal normalization to a tidy DataFrame.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests  # type: ignore[import-untyped]
from esports_quant.config import get_opendota_api_key
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from urllib3.util.retry import Retry  # type: ignore[import-untyped]

OPEN_DOTA_BASE = "https://api.opendota.com/api"
PRO_MATCHES_ENDPOINT = f"{OPEN_DOTA_BASE}/proMatches"
MATCH_DETAIL_ENDPOINT = f"{OPEN_DOTA_BASE}/matches"
CONSTANTS_ENDPOINT = f"{OPEN_DOTA_BASE}/constants"

# Premium allows 1200/min; we'll be conservative.
MIN_INTERVAL_S = 0.05

RETRY_TOTAL = 5
RETRY_BACKOFF = 0.3
RETRY_STATUS = (429, 500, 502, 503, 504)


def _session_with_retries(total: int = RETRY_TOTAL, backoff: float = RETRY_BACKOFF) -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff,
        status_forcelist=RETRY_STATUS,
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "esports_model_project/0.1 (contact: Dylan McGuinness)"})
    return sess


def _inject_api_key(params: Dict[str, Any] | None) -> Dict[str, Any]:
    out = dict(params or {})
    key = get_opendota_api_key()
    if key:
        out["api_key"] = key
    return out


def fetch_pro_matches(limit: int = 1000, page_size: int = 100) -> pd.DataFrame:
    """
    Fetch up to `limit` most-recent pro matches via /proMatches pagination.
    Returns a normalized DataFrame.
    """
    sess = _session_with_retries()
    records: List[Dict[str, Any]] = []
    fetched = 0
    less_than: Optional[int] = None

    while fetched < limit:
        params: Dict[str, Any] = {}
        if less_than is not None:
            params["less_than_match_id"] = less_than
        params = _inject_api_key(params)

        resp = sess.get(PRO_MATCHES_ENDPOINT, params=params, timeout=20)
        resp.raise_for_status()
        page = resp.json()
        if not isinstance(page, list) or not page:
            break

        take = min(page_size, limit - fetched)
        records.extend(page[:take])
        fetched += take

        mids = [r.get("match_id") for r in page if isinstance(r.get("match_id"), int)]
        if not mids:
            break
        less_than = min(mids)

        time.sleep(MIN_INTERVAL_S)

    return _normalize_pro_matches(records[:limit])


def fetch_match_detail(match_id: int) -> Dict[str, Any]:
    sess = _session_with_retries()
    params = _inject_api_key({})
    resp = sess.get(f"{MATCH_DETAIL_ENDPOINT}/{int(match_id)}", params=params, timeout=25)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected detail payload for match_id={match_id}")
    time.sleep(MIN_INTERVAL_S)
    return data


def fetch_constants(name: str) -> Any:
    """
    Fetch constants blob (e.g., 'patch', 'heroes', etc.).
    NB: The shape may be OBJECT or ARRAY depending on resource.
    """
    sess = _session_with_retries()
    params = _inject_api_key({})
    resp = sess.get(f"{CONSTANTS_ENDPOINT}/{name}", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    time.sleep(MIN_INTERVAL_S)
    return data


def _normalize_pro_matches(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize selected /proMatches fields into a tidy DataFrame.

    Columns:
      match_id (int64), team_a_id (int64), team_b_id (int64), team_a_win (int8),
      start_time (datetime UTC), duration (int32), league_name (string),
      series_type (int16), bo (int16)
    """
    df = pd.DataFrame.from_records(list(records))
    if df.empty:
        return pd.DataFrame(
            {
                "match_id": pd.Series(dtype="int64"),
                "team_a_id": pd.Series(dtype="int64"),
                "team_b_id": pd.Series(dtype="int64"),
                "team_a_win": pd.Series(dtype="int8"),
                "start_time": pd.Series(dtype="datetime64[ns, UTC]"),
                "duration": pd.Series(dtype="int32"),
                "league_name": pd.Series(dtype="string"),
                "series_type": pd.Series(dtype="int16"),
                "bo": pd.Series(dtype="int16"),
            }
        )

    keep = [
        "match_id",
        "radiant_team_id",
        "dire_team_id",
        "radiant_win",
        "start_time",
        "duration",
        "league_name",
        "series_type",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.dropna(subset=["match_id", "radiant_team_id", "dire_team_id", "radiant_win"])

    # Use int64 to avoid overflow; keep positive IDs only.
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("int64")
    df = df[df["match_id"] > 0]  # guard against negatives

    df["radiant_team_id"] = pd.to_numeric(df["radiant_team_id"], errors="coerce").astype("int64")
    df["dire_team_id"] = pd.to_numeric(df["dire_team_id"], errors="coerce").astype("int64")
    df["radiant_win"] = pd.to_numeric(df["radiant_win"], errors="coerce").fillna(0).astype("int8")
    df["start_time"] = pd.to_datetime(df["start_time"], unit="s", utc=True)
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype("int32")
    else:
        df["duration"] = pd.Series(0, index=df.index, dtype="int32")
    if "league_name" in df.columns:
        df["league_name"] = df["league_name"].astype("string")
    else:
        df["league_name"] = pd.Series(pd.NA, index=df.index, dtype="string")

    st = pd.to_numeric(df.get("series_type", pd.Series([1] * len(df))), errors="coerce").fillna(1).astype("int16")
    df["series_type"] = st
    ser_map: Dict[int, int] = {0: 1, 1: 3, 2: 5, 3: 3}
    df["bo"] = df["series_type"].map(ser_map).fillna(3).astype("int16")

    df = df.rename(columns={"radiant_team_id": "team_a_id", "dire_team_id": "team_b_id"})
    df["team_a_win"] = df["radiant_win"].astype("int8")
    df = df.drop(columns=["radiant_win"], errors="ignore")

    df = (
        df[
            [
                "match_id",
                "team_a_id",
                "team_b_id",
                "team_a_win",
                "start_time",
                "duration",
                "league_name",
                "series_type",
                "bo",
            ]
        ]
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    return df
