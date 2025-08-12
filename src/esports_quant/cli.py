# Purpose: Typer CLI for dummy ingest/train/evaluate and OpenDota ingest+Elo.

from __future__ import annotations  # enable future typing

import json  # pretty print metrics
from pathlib import Path  # path handling
from typing import Any, cast  # casting for loaded model

import pandas as pd  # parquet IO
import typer  # CLI

from .data.dummy import ingest_to_parquet, make_dummy_matches  # dummy generator
from .data.opendota import ingest_opendota_to_parquet  # raw OpenDota→Parquet
from .features.elo import build_elo_features  # Elo features
from .models.baseline import train_baseline  # training
from .utils.io import load_pickle  # load model artifact

# Create the CLI app with a short help string
app = typer.Typer(help="Esports Quant CLI — minimal pipeline + OpenDota ingest")


@app.command()
def ingest(out_dir: str = typer.Option("data/processed", help="Output directory")) -> str:  # noqa: B008
    # Generate deterministic dummy dataset and write Parquet
    out_path = ingest_to_parquet(out_dir=out_dir)
    # Print feedback for the user
    typer.echo(f"Wrote: {out_path}")
    # Return the path (useful in tests)
    return str(out_path)


@app.command()
def ingest_opendota(
    limit: int = typer.Option(2000, help="How many recent pro matches to fetch"),  # noqa: B008
    raw_dir: str = typer.Option("data/raw", help="Where to store raw normalized Parquet"),  # noqa: B008
    processed_path: Path = typer.Option(  # noqa: B008
        Path("data/processed/matches.parquet"),
        help="Output processed Parquet path",
    ),
) -> None:
    # 1) Fetch normalized raw matches to Parquet
    raw_path = ingest_opendota_to_parquet(limit=limit, out_dir=raw_dir)
    typer.echo(f"Raw written: {raw_path}")
    # 2) Load raw data and compute Elo features
    df_raw = pd.read_parquet(raw_path)
    df_feat = build_elo_features(df_raw)
    # 3) Ensure output directory exists and write processed dataset
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(processed_path, index=False)
    typer.echo(f"Processed written: {processed_path}")


@app.command()
def train(
    data_path: str = typer.Option("data/processed/matches.parquet", help="Parquet path"),  # noqa: B008
    artifacts_dir: str = typer.Option("artifacts", help="Artifacts directory"),  # noqa: B008
) -> None:
    # Load dataset
    df = pd.read_parquet(data_path)
    # Train baseline and compute validation metrics
    result = train_baseline(df, artifacts_dir=artifacts_dir)
    # Report artifact path and metrics
    typer.echo(f"Model saved to: {result.model_path}")
    typer.echo(json.dumps(result.metrics, indent=2))


@app.command()
def evaluate(
    artifacts_dir: str = typer.Option("artifacts", help="Artifacts directory"),  # noqa: B008
    seed: int = typer.Option(99, help="Seed for fresh eval data"),  # noqa: B008
) -> None:
    # Create a fresh synthetic dataset for a quick sanity evaluation
    df = make_dummy_matches(seed=seed)
    # Extract features and labels
    X = df[["rating_diff", "bo"]].values
    y_true = df["team_a_win"].values
    # Load the persisted model; cast to Any so mypy doesn't complain about attributes
    model = cast(Any, load_pickle(Path(artifacts_dir) / "baseline_logit.pkl"))
    # Predict probabilities for the positive class
    y_prob = model.predict_proba(X)[:, 1]
    # Compute metrics (local import avoids potential circular imports at module load)
    from .evaluate.metrics import compute_core_metrics

    # Calculate and print metrics as JSON
    metrics = compute_core_metrics(y_true, y_prob)
    typer.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    # Allow `python -m esports_quant.cli ...`
    app()
