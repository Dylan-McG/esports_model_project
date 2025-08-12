# Purpose: Pull recent Dota 2 pro matches from OpenDota with robust HTTP retries,
# normalize to a tidy table, and write Parquet. Also compute a Bo value from series_type.

from __future__ import annotations  # future typing behavior

import time  # polite sleeps during pagination
from pathlib import Path  # filesystem paths
from typing import Any  # typing help

import pandas as pd  # tabular wrangling
import requests  # type: ignore[import-untyped]  # HTTP client (ignore missing type stubs)
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]  # mount retries
from urllib3.util.retry import Retry  # type: ignore[import-untyped]  # retry strategy

from ..utils.io import ensure_dir  # ensure directories

# Base URL for OpenDota
OPEN_DOTA_BASE = "https://api.opendota.com/api"
# Endpoint for recent pro matches
PRO_MATCHES_ENDPOINT = f"{OPEN_DOTA_BASE}/proMatches"


def _session_with_retries(total: int = 5, backoff: float = 0.3) -> requests.Session:
    """
    Build a requests.Session with retry/backoff for transient errors.
    """
    s = requests.Session()  # connection pooling
    retry = Retry(
        total=total,  # total attempts
        read=total,  # retry read errors
        connect=total,  # retry connect errors
        backoff_factor=backoff,  # exponential backoff starting at backoff
        status_forcelist=(429, 500, 502, 503, 504),  # typical transient HTTPs
        allowed_methods=("GET",),  # idempotent
        raise_on_status=False,  # don't raise for HTTP 4xx/5xx automatically
    )
    adapter = HTTPAdapter(max_retries=retry)  # adapter with retry policy
    s.mount("http://", adapter)  # mount for http
    s.mount("https://", adapter)  # mount for https
    s.headers.update(  # polite UA
        {"User-Agent": "esports_model_project/0.1 (contact: Dylan McGuinness)"}
    )
    return s  # return configured session


def fetch_pro_matches(limit: int = 1000, page_size: int = 100) -> pd.DataFrame:
    """
    Fetch `limit` most recent pro matches from /proMatches, paginating via less_than_match_id.
    Normalize a minimal set of columns needed downstream.
    """
    sess = _session_with_retries()  # session with retries
    records: list[dict[str, Any]] = []  # collected page rows
    fetched = 0  # counter
    less_than: int | None = None  # pagination cursor (exclusive upper bound)

    # Page until we reach `limit` or get an empty page
    while fetched < limit:
        params = {"less_than_match_id": less_than} if less_than is not None else {}  # cursor
        resp = sess.get(PRO_MATCHES_ENDPOINT, params=params, timeout=20)  # request page
        resp.raise_for_status()  # raise if network/HTTP error
        page = resp.json()  # parse JSON list
        if not page:  # stop if server returns empty page
            break
        take = min(page_size, limit - fetched)  # bound by limit
        records.extend(page[:take])  # append slice
        fetched += take  # bump counter
        less_than = min(int(m["match_id"]) for m in page)  # new cursor is min match_id in page
        time.sleep(0.1)  # polite delay

    df = pd.DataFrame.from_records(records)  # to DataFrame

    # Keep only what we need downstream
    keep = [
        "match_id",
        "radiant_team_id",
        "dire_team_id",
        "radiant_win",
        "start_time",
        "duration",
        "league_name",
        "series_type",  # 0=Bo1, 1=Bo3, 2=Bo5 (but can be others in the wild)
    ]
    df = df[keep].dropna(
        subset=["match_id", "radiant_team_id", "dire_team_id", "radiant_win"]
    )  # ensure teams+target present

    # Basic dtypes
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("int64")  # ids
    df["radiant_team_id"] = pd.to_numeric(df["radiant_team_id"], errors="coerce").astype("int64")
    df["dire_team_id"] = pd.to_numeric(df["dire_team_id"], errors="coerce").astype("int64")
    df["radiant_win"] = pd.to_numeric(df["radiant_win"], errors="coerce").fillna(0).astype(int)
    df["start_time"] = pd.to_datetime(df["start_time"], unit="s", utc=True)  # ts
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0).astype("int32")  # s

    # series_type is messy in the wild. Normalize to int codes first.
    st = pd.to_numeric(df["series_type"], errors="coerce")  # may contain NaN or unexpected ints
    st = st.fillna(1).astype("int16")  # default to 1 (treat as Bo3 if unknown)
    df["series_type"] = st  # assign back

    # Map series_type → best-of with a safe default
    ser_map: dict[int, int] = {0: 1, 1: 3, 2: 5, 3: 3}  # include 3→Bo3 seen in-the-wild
    df["bo"] = df["series_type"].map(ser_map).fillna(3).astype("int16")  # default Bo3 for unknowns

    # Rename for our schema: team A = radiant, team B = dire
    df = df.rename(
        columns={"radiant_team_id": "team_a_id", "dire_team_id": "team_b_id"}
    )  # consistent names
    df["team_a_win"] = df["radiant_win"].astype(int)  # binary target

    return df.reset_index(drop=True)  # clean index


def ingest_opendota_to_parquet(
    limit: int = 2000,
    out_dir: str | Path = "data/raw",
    out_name: str = "opendota_pro.parquet",
) -> Path:
    """
    Fetch + normalize pro matches → Parquet at out_dir/out_name. Return the Path written.
    """
    out_dir = ensure_dir(out_dir)  # ensure directory exists
    df = fetch_pro_matches(limit=limit)  # pull and normalize
    out_path = out_dir / out_name  # compose path
    df.to_parquet(out_path, index=False)  # write parquet
    return out_path  # for downstream use
