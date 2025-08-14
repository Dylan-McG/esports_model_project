# Esports Quantitative Modeling Project

This project ingests, processes, and models competitive esports data (currently focused on Dota 2 via the [OpenDota API](https://docs.opendota.com/)). It provides:

- **Data ingestion** scripts that fetch match, player, and event details.
- **Feature engineering** utilities to build match/team-level datasets for modeling.
- **Model training** and evaluation pipelines.
- **Calibration and backtesting** tools for probability estimates.
- **Lightweight CLI interface** for running all major tasks.

---

## Quickstart (Windows PowerShell)

```powershell
# 1) Install dependencies
poetry install

# 2) Ingest + cache + build tidy parquet tables
poetry run python -m esports_quant.ingest.opendota_details --limit-ids 500 --players

# 3) Generate the sanity CSV pack (quick “what’s in the box” snapshot)
poetry run python scripts/sanity_report.py

# 4) Build a minimal per-match feature frame
poetry run python -m esports_quant.cli build-features

## CLI Reference

**`ingest-opendota-details`**
Fetches pro match IDs, caches `/matches/{id}` JSON, and writes tidy Parquet tables:
`matches.parquet`, `match_picks.parquet`, `match_players.parquet`, plus event tables
(`events_objectives.parquet`, `events_kills.parquet`, `events_items.parquet`,
`events_abilities.parquet`, `events_wards.parquet`), `teamfights.parquet`, and `players_timeseries.parquet`.

**`build-features`**
Loads processed tables and writes a per-match feature frame to
`data/processed/features.parquet` (includes Elo diff proxy, draft counts, momentum features).

**`train`**
Trains the baseline model, saving under `artifacts/`.

**`evaluate`**
Evaluates the current model (log loss, Brier, etc.).

**`calibrate`**
Calibrates model probabilities (Platt / isotonic) and writes:
`artifacts/calibrated.pkl`, `artifacts/calibration_metrics.json`, and an optional reliability plot.

---

## Data Dictionary (brief)

### `matches.parquet` — one row per match
- `match_id` — OpenDota match id
- `start_time` / `start_time_unix` — UTC start time
- `patch` — semantic patch tag (e.g., `7.39`)
- `leagueid`, `league_name`, `series_id`, `series_type`, `bo` — series metadata
- `radiant_team_id`, `radiant_name`, `dire_team_id`, `dire_name`
- `radiant_score`, `dire_score`, `radiant_win` (bool/int)
- `rating_diff` — baseline Elo-ish diff (placeholder; can be replaced)
- **Momentum snapshots:** `gold_adv_10`, `gold_adv_20`, `gold_adv_30`
- **Momentum slopes:** `gold_slope_0_10`, `gold_slope_10_25`, `gold_slope_25_end`
- `comeback_flag` — simple early-lead reversal flag

### `match_picks.parquet` — draft events
- `match_id`, `is_pick` (1/0), `hero_id`, `team` (0=radiant, 1=dire), `order`

### `match_players.parquet` — per player
- `match_id`, `account_id`, `player_slot`, `hero_id`, `isRadiant`, `win`
- Core stats: `kills`, `deaths`, `assists`, `gold_per_min`, `xp_per_min`, `tower_damage`, `stuns`
- Lane info: `lane`, `lane_role`, `is_roaming`, `lane_efficiency_pct`, `rank_tier`

### Event tables — long-form
- `events_objectives.parquet` — normalized timeline events (rosh/towers/etc.)
- `events_kills.parquet` — kill logs
- `events_items.parquet` — purchase logs
- `events_abilities.parquet` — ability upgrades/casts
- `events_wards.parquet` — obs/sentry placements
- `teamfights.parquet` — per-fight aggregates (damage, deaths)
- `players_timeseries.parquet` — per-minute gold/xp/last_hits/denies

---

## Environment & API Key

1. Duplicate the example env file:
   ```powershell
   Copy-Item .env.example .env

## Edit .env and set your key:

OPENDOTA_API_KEY=YOUR_KEY

**Notes:**

- `.env` is **git-ignored**; your key will not be committed.
- The ingest script accepts both query-param and bearer header; either works with the same value.

---

## Development Notes

- Heavy data folders (`data/raw/`, `data/processed/`, `artifacts/`) are ignored via `.gitignore`.
- Small `README.md` placeholders in data folders are kept so the structure exists in git.
- Run `poetry run pre-commit run -a` before commits to apply linting and type checks.
- For a full rebuild: delete `data/processed/` and rerun `ingest-opendota-details`.

---

## License

This project is intended for educational and research purposes only.
Check [OpenDota API Terms](https://docs.opendota.com/) before using in production.
