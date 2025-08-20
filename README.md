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

## Sanity Report & Dataset Validation

We have added a **comprehensive data sanity check and descriptive reporting script** to validate processed datasets and produce quick-to-explore summary files.

---

### Script Location
scripts/sanity_report.py


---

### Purpose
- Verify that **all processed parquet files** are present and contain data.
- Print a quick **footprint summary** of row counts across all datasets.
- Generate **small sample CSVs** for manual inspection.
- Produce **descriptive statistics** (counts, medians, quantiles) that give an immediate sense of dataset health and shape.
- Serve as the **single entry point** for running all dataset validation checks.

---

### Usage
Run from the project root:
```bash
poetry run python scripts/sanity_report.py

No arguments are required.
The script will automatically locate files in:

data/processed/

### Example Output:

[ok] loaded matches.parquet: 517 rows
[ok] loaded match_picks.parquet: 12,229 rows
...
[footprint]
{
  "matches_rows": 517,
  "picks_rows": 12229,
  ...
}
  [wrote] data\processed\sanity\sample_matches_head.csv
  [wrote] data\processed\sanity\matches_by_patch.csv
  ...
[done] sanity report CSVs written under data/processed/sanity/

# Output CSVs

Generated in: data/processed/sanity/

File	Description
sample_matches_head.csv	First few rows of matches.parquet
matches_by_patch.csv	Count of matches grouped by patch
patch_momentum_summary.csv	Gold advantage momentum statistics per patch
draft_hero_wr_sample.csv	Hero winrate sample by draft order
timing_sample.csv	Example timing events per match
timing_quantiles.csv	Quantile statistics for event timings
item_first_times_sample.csv	First purchase timings for core items
item_hero_median_times.csv	Median purchase time per hero & item
ability_first10_sample.csv	First 10 ability upgrades per hero
minute10_hero_side_means.csv	Average stats per hero & side at 10 min
event_coverage_sample.csv	Sample showing coverage across event types

## Hero Name Handling

The script ensures hero_name exists in the match picks dataset before performing winrate or draft grouping.
If missing, it maps from hero_id using the load_hero_id_to_name() helper.

## Typical Workflow

After ingesting and processing OpenDota data:

# 1. Ingest basic match list
poetry run python -m esports_quant.cli ingest-opendota --limit 500

# 2️. Ingest full match details (rich events)
poetry run python -m esports_quant.cli ingest-opendota-details --limit-ids 500 --players

# 3️. Run sanity checks & descriptive stats
poetry run python scripts/sanity_report.py

## Repository Structure

esports_model_project/
│
├── data/
│   ├── raw/                  # Raw ingested parquet files
│   ├── processed/            # Cleaned / transformed parquet files
│   │   └── sanity/           # Output of sanity_report.py
│
├── scripts/
│   ├── build_meta.py         # Builds meta statistics for modeling
│   └── sanity_report.py      # ✅ New: full dataset validation & metrics
│
├── src/esports_quant/
│   ├── ingest/               # OpenDota ingestion modules
│   ├── evaluate/             # Evaluation & metrics
│   ├── cli.py                # Command-line entrypoints
│   └── config.py             # Config management
│
├── pyproject.toml            # Poetry project definition
└── README.md                 # Project documentation

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
