from __future__ import annotations

"""
Build quick meta tables from processed data:
- Hero win rates per patch (CSV + Parquet)
- Patch momentum summaries (CSV + Parquet)
- Latest-patch top heroes snapshot (CSV)

Run:
    poetry run python scripts/build_meta.py
"""

import sys
from pathlib import Path

import pandas as pd

# Optional: pull hero names from OpenDota constants (uses our client + cached constants)
try:
    from esports_quant.data.opendota import fetch_constants
except Exception:
    fetch_constants = None  # script still works without names


PROC_DIR = Path("data/processed")
OUT_HERO_PATCH_PQ = PROC_DIR / "hero_patch_winrates.parquet"
OUT_HERO_PATCH_CSV = PROC_DIR / "hero_patch_winrates.csv"
OUT_MOMENTUM_PQ = PROC_DIR / "patch_momentum.parquet"
OUT_MOMENTUM_CSV = PROC_DIR / "patch_momentum.csv"
OUT_LATEST_TOP = PROC_DIR / "latest_patch_top_heroes.csv"


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load required processed tables."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    matches_path = PROC_DIR / "matches.parquet"
    picks_path = PROC_DIR / "match_picks.parquet"

    if not matches_path.exists():
        print(f"[ERROR] Missing {matches_path}. Run ingestion first.", file=sys.stderr)
        sys.exit(1)
    if not picks_path.exists():
        print(f"[ERROR] Missing {picks_path}. Run ingestion first.", file=sys.stderr)
        sys.exit(1)

    matches = pd.read_parquet(matches_path)
    picks = pd.read_parquet(picks_path)
    return matches, picks


def hero_name_map() -> pd.DataFrame | None:
    """Return a small DataFrame mapping hero_id -> localized_name (if available)."""
    if fetch_constants is None:
        return None
    try:
        heroes = fetch_constants("heroes")  # usually a dict keyed by hero name
    except Exception:
        return None

    # Normalize into a DataFrame
    rows = []
    if isinstance(heroes, dict):
        it = heroes.values()
    elif isinstance(heroes, list):
        it = heroes
    else:
        return None

    for rec in it:
        if not isinstance(rec, dict):
            continue
        hid = rec.get("id")
        name = rec.get("localized_name") or rec.get("name")
        if hid is None:
            continue
        rows.append({"hero_id": int(hid), "hero_name": str(name) if name else None})
    return pd.DataFrame(rows).drop_duplicates("hero_id")


def build_hero_patch_winrates(matches: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """Compute pick counts, wins, win rate, and pick share per patch/hero."""
    # Use picks only (exclude bans)
    picks_only = picks[picks["is_pick"] == True].copy()  # noqa: E712

    # Avoid duplicate 'patch' columns on merge: drop any 'patch' in picks
    if "patch" in picks_only.columns:
        picks_only = picks_only.drop(columns=["patch"])

    # Join picks to match outcomes/patch
    mp = picks_only.merge(
        matches[["match_id", "team_a_win", "patch"]],
        on="match_id",
        how="inner",
    )

    # Radiant is team A; Dire is team B
    mp["hero_win"] = (
        ((mp["side"] == "radiant") & (mp["team_a_win"] == 1)) | ((mp["side"] == "dire") & (mp["team_a_win"] == 0))
    ).astype(int)

    hero_patch = (
        mp.groupby(["patch", "hero_id"], dropna=False)
        .agg(games=("hero_id", "size"), wins=("hero_win", "sum"))
        .reset_index()
    )
    hero_patch["win_rate"] = (hero_patch["wins"] / hero_patch["games"]).astype(float)

    # Pick share within patch
    patch_totals = hero_patch.groupby("patch", dropna=False)["games"].sum().rename("patch_games")
    hero_patch = hero_patch.merge(patch_totals, on="patch", how="left")
    hero_patch["pick_share_pct"] = 100.0 * hero_patch["games"] / hero_patch["patch_games"]

    # Attach hero names if available
    names = hero_name_map()
    if names is not None and not names.empty:
        hero_patch = hero_patch.merge(names, on="hero_id", how="left")

    # Sort for convenience
    hero_patch = hero_patch.sort_values(["patch", "games", "win_rate"], ascending=[True, False, False])
    return hero_patch.reset_index(drop=True)


def build_patch_momentum(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarize gold-advantage features per patch."""
    mom_cols = [
        "gold_adv_10",
        "gold_adv_20",
        "gold_adv_30",
        "gold_slope_0_10",
        "gold_slope_10_25",
        "gold_slope_25_end",
        "comeback_flag",
    ]
    keep = [c for c in mom_cols if c in matches.columns]
    if not keep:
        return pd.DataFrame({"patch": matches["patch"].unique()}) if "patch" in matches.columns else pd.DataFrame()

    momentum = matches.groupby("patch", dropna=False)[keep].agg(["mean", "median", "count"]).reset_index()

    # Flatten MultiIndex columns
    momentum.columns = ["patch"] + [f"{c}_{stat}" for c, stat in momentum.columns.tolist()[1:]]
    return momentum.sort_values("patch").reset_index(drop=True)


def write_outputs(hero_patch: pd.DataFrame, momentum: pd.DataFrame) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    hero_patch.to_parquet(OUT_HERO_PATCH_PQ, index=False)
    hero_patch.to_csv(OUT_HERO_PATCH_CSV, index=False)
    momentum.to_parquet(OUT_MOMENTUM_PQ, index=False)
    momentum.to_csv(OUT_MOMENTUM_CSV, index=False)

    # Latest patch snapshot (top 30 by games, then win rate)
    latest_patch = None
    if "patch" in hero_patch.columns and not hero_patch["patch"].dropna().empty:
        latest_patch = hero_patch["patch"].astype(str).dropna().max()

    if latest_patch is not None:
        latest = hero_patch[hero_patch["patch"].astype(str) == latest_patch].copy()
        latest_top = latest.sort_values(["games", "win_rate"], ascending=[False, False]).head(30)
        latest_top.to_csv(OUT_LATEST_TOP, index=False)

    print("[OK] Wrote:")
    print(f"  - {OUT_HERO_PATCH_CSV}")
    print(f"  - {OUT_MOMENTUM_CSV}")
    if latest_patch is not None:
        print(f"  - {OUT_LATEST_TOP} (latest patch: {latest_patch})")
    print("  (Parquet equivalents written too.)")

    # Small console preview
    print("\n[Sample] hero_patch_winrates:")
    cols = [
        c
        for c in ["patch", "hero_id", "hero_name", "games", "wins", "win_rate", "pick_share_pct"]
        if c in hero_patch.columns
    ]
    print(hero_patch[cols].head(12).to_string(index=False))

    if not momentum.empty:
        print("\n[Sample] patch_momentum:")
        print(momentum.head(6).to_string(index=False))


def main() -> None:
    matches, picks = load_processed()
    hero_patch = build_hero_patch_winrates(matches, picks)
    momentum = build_patch_momentum(matches)
    write_outputs(hero_patch, momentum)


if __name__ == "__main__":
    main()
