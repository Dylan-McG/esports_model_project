# Purpose: Pull recent Dota 2 pro matches from OpenDota with robust HTTP retries,
# normalize to a tidy table, and write Parquet. Also compute a Bo value from series_type.

from __future__ import annotations  # future typing behavior

import time  # polite sleeps during pagination
from pathlib import Path  # filesystem paths

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
    # Create a session for connection reuse and retry control
    s = requests.Session()
    # Configure transient-error retries with exponential backoff
    retry = Retry(
        total=total,
        read=total,
        connect=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    # Attach the retry policy to HTTP and HTTPS
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    # Identify the client politely
    s.headers.update({"User-Agent": "esports_model_project/0.1 (contact: Dylan McGuinness)"})
    # Return the configured session
    return s


def fetch_pro_matches(limit: int = 1000, page_size: int = 100) -> pd.DataFrame:
    # Fetch `limit` most recent pro matches with pagination via less_than_match_id
    sess = _session_with_retries()
    records: list[dict] = []
    fetched = 0
    less_than: int | None = None

    # Loop until we have enough rows or run out of pages
    while fetched < limit:
        # Build query params for pagination (most recent first → page backward)
        params = {"less_than_match_id": less_than} if less_than is not None else {}
        # Request a page
        resp = sess.get(PRO_MATCHES_ENDPOINT, params=params, timeout=20)
        resp.raise_for_status()
        page = resp.json()
        # Stop if server returns empty page
        if not page:
            break
        # Take up to page_size but not beyond the requested limit
        take = min(page_size, limit - fetched)
        records.extend(page[:take])
        fetched += take
        # Update cursor to the smallest match_id seen to paginate backwards
        less_than = min(int(m["match_id"]) for m in page)
        # Be polite with a tiny delay
        time.sleep(0.1)

    # Convert to DataFrame
    df = pd.DataFrame.from_records(records)

    # Keep only columns we need downstream
    keep = [
        "match_id",
        "radiant_team_id",
        "dire_team_id",
        "radiant_win",
        "start_time",
        "duration",
        "league_name",
        "series_type",  # 0=Bo1, 1=Bo3, 2=Bo5
    ]
    df = df[keep].dropna(
        subset=[
            "match_id",
            "radiant_team_id",
            "dire_team_id",
            "radiant_win",
        ]
    )

    # Fix dtypes
    df["match_id"] = df["match_id"].astype("int64")
    df["radiant_team_id"] = df["radiant_team_id"].astype("int64")
    df["dire_team_id"] = df["dire_team_id"].astype("int64")
    df["radiant_win"] = df["radiant_win"].astype(int)
    df["start_time"] = pd.to_datetime(df["start_time"], unit="s", utc=True)
    df["duration"] = df["duration"].astype("int32")
    df["series_type"] = df["series_type"].fillna(0).astype("int16")

    # Map series_type to a best-of value
    ser_map = {0: 1, 1: 3, 2: 5}
    df["bo"] = df["series_type"].map(ser_map).astype("int16")

    # Rename for our schema: team A = radiant, team B = dire
    df = df.rename(columns={"radiant_team_id": "team_a_id", "dire_team_id": "team_b_id"})
    # Target variable
    df["team_a_win"] = df["radiant_win"].astype(int)

    # Return normalized frame
    return df.reset_index(drop=True)


def ingest_opendota_to_parquet(
    limit: int = 2000,
    out_dir: str | Path = "data/raw",
    out_name: str = "opendota_pro.parquet",
) -> Path:
    # Ensure output directory exists
    out_dir = ensure_dir(out_dir)
    # Fetch matches
    df = fetch_pro_matches(limit=limit)
    # Compose output path
    out_path = out_dir / out_name
    # Write Parquet
    df.to_parquet(out_path, index=False)
    # Return path for downstream use
    return out_path
