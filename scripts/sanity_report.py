# scripts/sanity_report.py
from __future__ import annotations

import itertools as it
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data" / "processed"
OUT = PROC / "sanity"
OUT.mkdir(parents=True, exist_ok=True)


def exists(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def load_if(path: Path, cols: list[str] | None = None) -> pd.DataFrame | None:
    if not exists(path):
        print(f"[skip] missing {path.name}")
        return None
    try:
        df = pd.read_parquet(path, columns=cols)
        print(f"[ok] loaded {path.name}: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"[warn] failed to read {path.name}: {e}")
        return None


# ---- hero_name helpers (no client import) ----
def load_hero_id_to_name_from_cache() -> dict[int, str] | None:
    cache_path = Path("data/raw/constants/heroes.json")
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        id2name: dict[int, str] = {}
        for _, meta in (data or {}).items():
            try:
                hid = int(meta.get("id"))
            except (TypeError, ValueError):
                continue
            name = meta.get("localized_name") or meta.get("name") or f"hero_{hid}"
            id2name[hid] = name
        return id2name
    except Exception:
        return None


def infer_id2name_from_frames(players: pd.DataFrame | None, picks: pd.DataFrame | None) -> dict[int, str] | None:
    pieces = []
    if players is not None and {"hero_id", "hero_name"} <= set(players.columns):
        pieces.append(players[["hero_id", "hero_name"]])
    if picks is not None and {"hero_id", "hero_name"} <= set(picks.columns):
        pieces.append(picks[["hero_id", "hero_name"]])
    if not pieces:
        return None
    df = pd.concat(pieces, ignore_index=True).dropna().drop_duplicates()
    try:
        return {int(h): str(n) for h, n in zip(df["hero_id"], df["hero_name"], strict=False)}
    except Exception:
        return None


def ensure_hero_name(df: pd.DataFrame, id2name: dict[int, str] | None, id_col="hero_id", out_col="hero_name"):
    if df is None or id_col not in df.columns:
        return df
    if out_col not in df.columns:
        df[out_col] = pd.NA
    if id2name:
        df[out_col] = df[out_col].fillna(df[id_col].map(id2name))
    mask = df[out_col].isna()
    if mask.any():
        df.loc[mask, out_col] = "hero_" + df.loc[mask, id_col].astype("Int64").astype(str)
    df[out_col] = df[out_col].astype("string")
    return df


# ---- player flags helpers ----
def ensure_player_flags(players: pd.DataFrame | None, matches: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Ensure `is_radiant` and `win` exist in player-level table:
      - is_radiant: from player_slot (<128) or team (0 Radiant/1 Dire)
      - win: from is_radiant + matches.radiant_win
    """
    if players is None:
        return None
    df = players.copy()

    # is_radiant
    if "is_radiant" not in df.columns:
        if "player_slot" in df.columns:
            df["is_radiant"] = df["player_slot"].astype("Int64") < 128
        elif "team" in df.columns:
            df["is_radiant"] = df["team"].map({0: True, 1: False})
        else:
            df["is_radiant"] = pd.NA

    # win
    if "win" not in df.columns:
        if matches is not None and "radiant_win" in matches.columns and "match_id" in df.columns:
            df = df.merge(matches[["match_id", "radiant_win"]], on="match_id", how="left")
            df["win"] = np.where(
                df["is_radiant"] == True,
                df["radiant_win"] == True,
                np.where(df["is_radiant"] == False, df["radiant_win"] == False, pd.NA),
            )
            df = df.drop(columns=["radiant_win"])
        else:
            df["win"] = pd.NA

    return df


def save_csv(df: pd.DataFrame, name: str, index=False):
    out = OUT / name
    df.to_csv(out, index=index)
    print(f"  [wrote] {out.relative_to(BASE)}")


def main():
    # ---------- Load what we can ----------
    matches = load_if(PROC / "matches.parquet")
    picks = load_if(PROC / "match_picks.parquet")
    players = load_if(PROC / "match_players.parquet")
    ev_obj = load_if(PROC / "events_objectives.parquet")
    ev_kill = load_if(PROC / "events_kills.parquet")
    ev_items = load_if(PROC / "events_items.parquet")
    ev_abil = load_if(PROC / "events_abilities.parquet")
    ev_wards = load_if(PROC / "events_wards.parquet")
    teamfights = load_if(PROC / "teamfights.parquet")
    timeseries = load_if(PROC / "players_timeseries.parquet")

    # Build hero id->name mapping
    id2name = load_hero_id_to_name_from_cache() or infer_id2name_from_frames(players, picks) or {}

    # Normalize hero_name presence
    if picks is not None:
        picks = ensure_hero_name(picks, id2name)
    if players is not None:
        players = ensure_hero_name(players, id2name)

    # Ensure player-side/win flags exist
    players = ensure_player_flags(players, matches)

    # ---------- Dataset footprint ----------
    footprint = {
        "matches_rows": len(matches) if matches is not None else 0,
        "picks_rows": len(picks) if picks is not None else 0,
        "players_rows": len(players) if players is not None else 0,
        "events_objectives_rows": len(ev_obj) if ev_obj is not None else 0,
        "events_kills_rows": len(ev_kill) if ev_kill is not None else 0,
        "events_items_rows": len(ev_items) if ev_items is not None else 0,
        "events_abilities_rows": len(ev_abil) if ev_abil is not None else 0,
        "events_wards_rows": len(ev_wards) if ev_wards is not None else 0,
        "teamfights_rows": len(teamfights) if teamfights is not None else 0,
        "players_timeseries_rows": len(timeseries) if timeseries is not None else 0,
    }
    (OUT / "footprint.json").write_text(json.dumps(footprint, indent=2))
    print("\n[footprint]")
    print(json.dumps(footprint, indent=2))

    # ---------- Matches quick look ----------
    if matches is not None:
        basic_cols = [
            c
            for c in [
                "match_id",
                "start_time",
                "bo",
                "rating_diff",
                "patch",
                "gold_adv_10",
                "gold_adv_20",
                "gold_adv_30",
                "comeback_flag",
            ]
            if c in matches.columns
        ]
        save_csv(matches[basic_cols].head(20), "sample_matches_head.csv")

        agg_cols = ["patch"] if "patch" in matches.columns else []
        g = matches.groupby(agg_cols) if agg_cols else matches.assign(_=0).groupby("_")
        sums = g.agg(
            matches=("match_id", "count"),
            duration_median=("duration", "median") if "duration" in matches.columns else ("match_id", "count"),
        ).reset_index()
        save_csv(sums, "matches_by_patch.csv")

        ga_cols = [c for c in ["gold_adv_10", "gold_adv_20", "gold_adv_30"] if c in matches.columns]
        if ga_cols:
            if "patch" in matches.columns:
                mom = matches[["patch", *ga_cols]].groupby("patch", dropna=False).agg(["mean", "median", "count"])
                mom.columns = [f"{a}_{b}" for a, b in mom.columns]
                mom = mom.reset_index()
            else:
                mom = matches[ga_cols].agg(["mean", "median", "count"]).T.reset_index(names="metric")
            save_csv(mom, "patch_momentum_summary.csv")

    # ---------- Draft quick look ----------
    if picks is not None and matches is not None and "radiant_win" in matches.columns:
        mp = picks.merge(matches[["match_id", "patch", "radiant_win"]], on="match_id", how="left")
        # team==0 Radiant, team==1 Dire (OpenDota picks_bans)
        if "team" in mp.columns and "is_pick" in mp.columns:
            mp_pick = mp[mp["is_pick"] == True].copy()
            mp_pick["won"] = np.where(
                mp_pick["team"] == 0,
                mp_pick["radiant_win"] == True,
                mp_pick["radiant_win"] == False,
            )
            hero_wr = (
                mp_pick.groupby(["patch", "hero_id", "hero_name"], dropna=False)
                .agg(games=("match_id", "nunique"), wins=("won", "sum"))
                .reset_index()
            )
            hero_wr["win_rate"] = hero_wr["wins"] / hero_wr["games"]
            hero_wr = hero_wr.sort_values(["patch", "games"], ascending=[True, False])
            save_csv(hero_wr.head(200), "draft_hero_wr_sample.csv")

    # ---------- Event timings ----------
    fb = (
        (ev_kill.groupby("match_id", as_index=False)["time_s"].min().rename(columns={"time_s": "first_blood_s"}))
        if ev_kill is not None
        else None
    )

    if ev_obj is not None:
        rosh = (
            ev_obj[ev_obj["type"] == "CHAT_MESSAGE_ROSHAN_KILL"]
            .groupby("match_id", as_index=False)["time_s"]
            .min()
            .rename(columns={"time_s": "first_rosh_s"})
        )
        t1 = ev_obj[(ev_obj["type"] == "building_kill") & (ev_obj["key"].str.contains("tower1_", na=False))]
        first_t1 = t1.groupby("match_id", as_index=False)["time_s"].min().rename(columns={"time_s": "first_t1_down_s"})
    else:
        rosh = None
        first_t1 = None

    if matches is not None:
        timing = matches[["match_id", "patch"]].copy()
        for add in [fb, rosh, first_t1]:
            if add is not None:
                timing = timing.merge(add, on="match_id", how="left")
        save_csv(timing.head(100), "timing_sample.csv")
        timing_q = timing.drop(columns=["match_id"]).groupby("patch").quantile([0.25, 0.5, 0.75]).reset_index()
        save_csv(timing_q, "timing_quantiles.csv")

    # ---------- Item timings ----------
    if ev_items is not None and players is not None:
        key_items = {"blink", "black_king_bar", "guardian_greaves", "aghanims_scepter"}
        kk = ev_items[ev_items["item"].isin(key_items)].copy()
        first_item = kk.groupby(["match_id", "account_id", "item"], as_index=False)["time_s"].min()

        # only select columns that exist after ensuring flags
        want = ["match_id", "account_id", "hero_id", "hero_name", "is_radiant", "win"]
        have = [c for c in want if c in players.columns]
        joinp = players[have]
        pl_items = first_item.merge(joinp, on=["match_id", "account_id"], how="left")
        save_csv(pl_items.head(200), "item_first_times_sample.csv")

        if {"hero_name", "item"}.issubset(pl_items.columns):
            hero_item = (
                pl_items.groupby(["hero_name", "item"], as_index=False)["time_s"]
                .median()
                .rename(columns={"time_s": "time_s_median"})
            )
            save_csv(
                hero_item.sort_values(["item", "time_s_median"]).head(200),
                "item_hero_median_times.csv",
            )

    # ---------- Player-pair synergy ----------
    if players is not None and matches is not None and "win" in players.columns:
        mcols = [c for c in ["match_id", "patch"] if c in matches.columns]
        if "match_id" in mcols:
            m2 = players.merge(matches[mcols], on="match_id", how="left")
            if "account_id" in m2.columns and "is_radiant" in m2.columns:
                team_win = m2.groupby(["match_id", "is_radiant"], as_index=False)["win"].max()
                if "patch" in matches.columns:
                    team_win = team_win.merge(matches[["match_id", "patch"]], on="match_id", how="left")
                mp_small_cols = [c for c in ["match_id", "is_radiant", "patch", "account_id"] if c in m2.columns]
                mp_small = m2.dropna(subset=["account_id"])[mp_small_cols]
                pairs = (
                    mp_small.groupby([c for c in ["patch", "match_id", "is_radiant"] if c in mp_small.columns])[
                        "account_id"
                    ]
                    .apply(lambda s: list(it.combinations(sorted(set(s)), 2)))
                    .explode()
                    .dropna()
                )
                if len(pairs):
                    pairs = pairs.apply(pd.Series).rename(columns={0: "a", 1: "b"}).astype({"a": "int64", "b": "int64"})
                    merge_keys = [
                        k
                        for k in ["match_id", "is_radiant", "patch"]
                        if k in team_win.columns and k in mp_small.columns
                    ]
                    pairs = pairs.merge(team_win[merge_keys + ["win"]], on=merge_keys, how="left")
                    group_keys = [k for k in ["patch", "a", "b"] if k in pairs.columns]
                    pair_wr = (
                        pairs.groupby(group_keys)
                        .agg(games=("win", "size"), wr=("win", "mean"))
                        .reset_index()
                        .query("games >= 5")
                        .sort_values(group_keys + ["wr", "games"], ascending=[True, True, True, False, False])
                    )
                    save_csv(pair_wr.head(200), "pair_synergy_top.csv")

    # ---------- Ability early casts ----------
    if ev_abil is not None and players is not None:
        ab = ev_abil.sort_values(["match_id", "account_id", "time_s"])
        join_cols = [c for c in ["match_id", "account_id", "hero_name"] if c in players.columns]
        first10 = (
            ab.groupby(["match_id", "account_id"])
            .head(10)
            .merge(players[join_cols], on=["match_id", "account_id"], how="left")
        )
        save_csv(first10.head(200), "ability_first10_sample.csv")

        # ---------- Per‑player minute curves (minute 10 snapshot) ----------
    if timeseries is not None and "minute" in timeseries.columns:
        df10 = timeseries[timeseries["minute"] == 10].copy()

        # what metrics are available?
        value_cols = [c for c in ["lh", "gold", "xp"] if c in df10.columns]
        if value_cols:  # only proceed if at least one metric exists
            # prefer grouping by hero_name and side if present
            group_keys = [c for c in ["hero_name", "is_radiant"] if c in df10.columns]
            if not group_keys:
                # no group keys available → aggregate overall (use a dummy key)
                df10["_"] = 0
                group_keys = ["_"]

            snap10 = (
                df10.groupby(group_keys, as_index=False)[value_cols].mean().sort_values(value_cols[0], ascending=False)
            )
            save_csv(snap10.head(200), "minute10_hero_side_means.csv")

    # ---------- Event coverage per match ----------
    cover_parts = []
    for name, df in [
        ("objectives", ev_obj),
        ("kills", ev_kill),
        ("items", ev_items),
        ("abilities", ev_abil),
        ("wards", ev_wards),
        ("teamfights", teamfights),
    ]:
        if df is None or "match_id" not in (df.columns if df is not None else []):
            continue
        kk = df.groupby("match_id", as_index=False).size().rename(columns={"size": f"{name}_count"})
        cover_parts.append(kk)

    if cover_parts and matches is not None:
        cov = matches[[c for c in ["match_id", "patch"] if c in matches.columns]]
        for part in cover_parts:
            cov = cov.merge(part, on="match_id", how="left")
        save_csv(cov.head(200), "event_coverage_sample.csv")

    print("\n[done] sanity report CSVs written under data/processed/sanity/")


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    main()
