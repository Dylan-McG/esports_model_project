from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import joblib
import pandas as pd
import typer

from esports_quant.features.build_features import build_match_team_features
from esports_quant.evaluate.calibration import calibrate_and_save
from .ingest.opendota import ingest_pro_matches
from .ingest.opendota_details import BuildConfig, run_full_build
from .models.baseline import train_baseline

app = typer.Typer(help="Esports Quant CLI — ingest, build features, train, evaluate, calibrate, backtest")


# -----------------------------------------------------------------------------
# Build features
# -----------------------------------------------------------------------------
@app.command("build-features")
def cmd_build_features() -> None:
    """Build features and save to data/processed/features.parquet."""
    proc = Path("data/processed")
    matches = pd.read_parquet(proc / "matches.parquet")
    picks = pd.read_parquet(proc / "match_picks.parquet")
    players_path = proc / "match_players.parquet"
    players = pd.read_parquet(players_path) if players_path.exists() else None

    feats = build_match_team_features(matches, picks, players=players)
    out = proc / "features.parquet"
    feats.to_parquet(out, index=False)
    print(f"[OK] Wrote features ({len(feats)}) rows to {out}")


# -----------------------------------------------------------------------------
# Ingest OpenDota (quick path: proMatches → matches.parquet)
# -----------------------------------------------------------------------------
@app.command("ingest-opendota", help="Fetch pro matches, build Elo features, write processed Parquet.")
def cmd_ingest_opendota(
    limit: Annotated[int, typer.Option(help="How many most-recent pro matches to fetch")] = 1000,
    processed_path: Annotated[Path, typer.Option(help="Output processed Parquet path")] = Path(
        "data/processed/matches.parquet"
    ),
) -> None:
    """Uses the light ingest path (summary records only)."""
    out = ingest_pro_matches(limit=limit, processed_path=processed_path)
    typer.echo(f"Wrote processed dataset to: {out}")


# -----------------------------------------------------------------------------
# Ingest OpenDota (details: cache + tidy + patch + timelines)
# -----------------------------------------------------------------------------
@app.command("ingest-opendota-details", help="Cache /proMatches + /matches/{id} and build tidy Parquets.")
def cmd_ingest_opendota_details(
    limit_ids: Annotated[
        Optional[int],
        typer.Option(help="Limit how many most-recent match IDs to process (None = all fetched)."),
    ] = 5000,
    include_players: Annotated[bool, typer.Option("--players", help="Also build match_players.parquet.")] = False,
    elo_k: Annotated[float, typer.Option(help="Elo K-factor for pre-match diff.")] = 20.0,
    elo_base: Annotated[float, typer.Option(help="Elo base rating.")] = 1500.0,
) -> None:
    """
    Full ingestion with raw caching and rich tidy outputs:
      - data/raw/proMatches/page_*.json (ID snapshots)
      - data/raw/matchDetails/{match_id}.json (detail cache)
      - data/processed/matches.parquet (match-level with patch/momentum/etc.)
      - data/processed/match_picks.parquet (draft picks/bans)
      - data/processed/match_players.parquet (optional, --players)
      - events_* / teamfights / players_timeseries parquet tables
    """
    cfg = BuildConfig(limit_ids=limit_ids, cache_players=include_players, elo_k=elo_k, elo_base=elo_base)
    report = run_full_build(cfg)
    typer.echo(json.dumps(report, indent=2))


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
@app.command()
def train(
    data_path: Annotated[Path, typer.Option(help="Processed Parquet path")] = Path("data/processed/matches.parquet"),
    artifacts_dir: Annotated[Path, typer.Option(help="Where to store models")] = Path("artifacts"),
) -> None:
    """Train baseline model and print quick (in-sample) metrics."""
    df = pd.read_parquet(data_path)
    out = train_baseline(df, artifacts_dir=artifacts_dir)
    typer.echo(f"Model saved to: {out}")

    # Quick metrics on train (smoke check; not a holdout)
    x = df[["rating_diff", "bo"]].to_numpy()
    y = df["team_a_win"].to_numpy()
    model = joblib.load(out)
    y_prob = model.predict_proba(x)[:, 1]

    from sklearn.metrics import brier_score_loss, log_loss

    metrics = {
        "log_loss": float(log_loss(y, y_prob)),
        "brier": float(brier_score_loss(y, y_prob)),
    }
    typer.echo(json.dumps(metrics, indent=2))


# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------
@app.command()
def evaluate(
    data_path: Annotated[Path, typer.Option(help="Processed Parquet path")] = Path("data/processed/matches.parquet"),
    model_path: Annotated[Path, typer.Option(help="Model to evaluate")] = Path("artifacts/baseline_logit.pkl"),
) -> None:
    """Evaluate a saved model on the given dataset."""
    from sklearn.metrics import brier_score_loss, log_loss

    df = pd.read_parquet(data_path)
    model = joblib.load(model_path)

    x = df[["rating_diff", "bo"]].to_numpy()
    y = df["team_a_win"].to_numpy()
    y_prob = model.predict_proba(x)[:, 1]

    metrics = {
        "log_loss": float(log_loss(y, y_prob)),
        "brier": float(brier_score_loss(y, y_prob)),
    }
    typer.echo(json.dumps(metrics, indent=2))


# -----------------------------------------------------------------------------
# Calibrate
# -----------------------------------------------------------------------------
@app.command(help="Fit a probability calibrator (Platt or Isotonic) on processed data.")
def calibrate(
    method: Annotated[
        str,
        typer.Option(help="Calibration method. One of: 'platt' or 'isotonic' (case-insensitive)."),
    ] = "platt",
    data_path: Annotated[Path, typer.Option(help="Processed Parquet path")] = Path("data/processed/matches.parquet"),
    out_model: Annotated[Path, typer.Option(help="Output calibrated model path")] = Path("artifacts/calibrated.pkl"),
    out_plot: Annotated[Path, typer.Option(help="Calibration plot path")] = Path("artifacts/calibration.png"),
    out_metrics: Annotated[Path, typer.Option(help="Calibration metrics JSON path")] = Path(
        "artifacts/calibration_metrics.json"
    ),
) -> None:
    """Calibrate probabilities and save artifacts."""
    m = method.strip().lower()
    if m not in {"platt", "isotonic"}:
        raise typer.BadParameter("method must be 'platt' or 'isotonic'")

    df = pd.read_parquet(data_path)
    metrics = calibrate_and_save(
        df=df,
        method=m,
        out_model=str(out_model),
        out_plot=str(out_plot),
        out_metrics=str(out_metrics),
    )
    typer.echo(json.dumps(metrics, indent=2))


# -----------------------------------------------------------------------------
# Backtest (rolling) — stub wiring; implement in evaluate/backtest.py
# -----------------------------------------------------------------------------
@app.command(help="Rolling-window backtest over time-sorted data.")
def backtest(
    data_path: Annotated[Path, typer.Option(help="Processed Parquet path")] = Path("data/processed/matches.parquet"),
    n_splits: Annotated[int, typer.Option(help="Number of folds")] = 5,
    min_train: Annotated[int, typer.Option(help="Min rows in first train window")] = 800,
    step: Annotated[int, typer.Option(help="Rows to advance per fold")] = 200,
    out_plot: Annotated[Path, typer.Option(help="Plot output path")] = Path("artifacts/backtest_metrics.png"),
    out_csv: Annotated[Path, typer.Option(help="CSV output path")] = Path("artifacts/backtest_metrics.csv"),
    datetime_col: Annotated[str, typer.Option(help="Timestamp column")] = "start_time",
) -> None:
    """Run a simple rolling backtest and write metrics/plot."""
    from .evaluate.backtest import rolling_backtest

    df = pd.read_parquet(data_path)
    out = rolling_backtest(
        df=df,
        datetime_col=datetime_col,
        n_splits=n_splits,
        min_train=min_train,
        step=step,
        out_plot=str(out_plot),
        out_csv=str(out_csv),
    )
    typer.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    app()
