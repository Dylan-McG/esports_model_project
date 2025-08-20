from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional
from typing import List

import joblib
import numpy as np
import pandas as pd
import typer

from esports_quant.evaluate.calibration import calibrate_and_save
from esports_quant.features.build_features import (
    build_pregame_features,  # H0: pre-game builder
    build_postdraft_features,  # H1: post-draft builder
)
from .ingest.opendota import ingest_pro_matches
from .ingest.opendota_details import BuildConfig, run_full_build
from .models.baseline import train_baseline

app = typer.Typer(help="Esports Quant CLI — ingest, build features, train, evaluate, calibrate, backtest")


# -----------------------------------------------------------------------------
# Helper — pick columns that match what the model expects
# -----------------------------------------------------------------------------
def _infer_expected_n_features(model) -> int | None:
    """Try to read how many features the pipeline was fit with."""
    try:
        imputer = getattr(model, "named_steps", {}).get("imputer", None)
        if imputer is not None:
            # statistics_.shape[0] equals number of features seen during fit
            if hasattr(imputer, "statistics_"):
                return int(imputer.statistics_.shape[0])
            if hasattr(imputer, "n_features_in_"):
                return int(imputer.n_features_in_)
    except Exception:
        pass
    return None


def _candidate_feature_sets(df: pd.DataFrame) -> list[list[str]]:
    """
    Return ordered candidate feature sets we know how to build, from most- to least-rich.
    """
    # Post-draft (H1) features (draft winrates) in a fixed, explicit order:
    draft_cols = [
        "rating_diff",
        "bo",
        "draft_wr_sum_radiant",
        "draft_wr_avg_radiant",
        "draft_wr_sum_dire",
        "draft_wr_avg_dire",
    ]
    # Pre-game (H0)
    pre_cols = ["rating_diff", "bo"]

    cands: list[list[str]] = []
    if all(c in df.columns for c in draft_cols):
        cands.append(draft_cols)
    if all(c in df.columns for c in pre_cols):
        cands.append(pre_cols)
    # Fall back order if neither is fully present (we’ll error later)
    return cands or [pre_cols]


def _get_feature_matrix_for_model(df: pd.DataFrame, model) -> tuple[np.ndarray, list[str]]:
    """
    Return X as a float numpy array in the exact feature order the model was trained on.
    Preference order:
      1) model._feature_cols if present
      2) match a known candidate column set whose length equals the model's expected n_features
      3) best-effort candidate (will error if lengths mismatch)
    """
    # 1) If the model advertises its columns, use them directly.
    feature_cols_attr = getattr(model, "_feature_cols", None)
    if feature_cols_attr:
        feature_cols = list(feature_cols_attr)
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise typer.BadParameter(f"Dataset missing model-required feature columns: {sorted(missing)}")
        return df[feature_cols].to_numpy(dtype=float), feature_cols

    # 2) Otherwise, infer how many features the trained pipeline expects
    n_expected = _infer_expected_n_features(model)

    # 3) Try to select a candidate set that both exists in df and matches n_expected
    for cols in _candidate_feature_sets(df):
        if (n_expected is None) or (len(cols) == n_expected):
            X = df[cols].to_numpy(dtype=float)
            return X, cols

    # 4) If none matched length exactly, provide a helpful error
    avail = [c for c in df.columns]
    raise typer.BadParameter(
        "Could not align feature columns with the trained model.\n"
        f"Model expects {n_expected} features, but available known sets don't match.\n"
        "Available columns include (first 50): " + ", ".join(avail[:50])
    )


# -----------------------------------------------------------------------------
# Build features — PRE-GAME (H0)
# -----------------------------------------------------------------------------
@app.command("build-features-pre")
def cmd_build_features_pre(
    matches_path: Annotated[Path, typer.Option(help="Path to matches.parquet")] = Path(
        "data/processed/matches.parquet"
    ),
    out_path: Annotated[Path, typer.Option(help="Output path for pre-game features")] = Path(
        "data/processed/features_pre.parquet"
    ),
) -> None:
    """Build pre-game (H0) features and save to data/processed/features_pre.parquet."""
    matches = pd.read_parquet(matches_path)
    feats_pre = build_pregame_features(matches)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats_pre.to_parquet(out_path, index=False)
    typer.echo(f"[OK] Wrote pre-game features ({len(feats_pre)}) rows to {out_path}")


# -----------------------------------------------------------------------------
# Build features — POST-DRAFT (H1)
# -----------------------------------------------------------------------------
@app.command("build-features-draft")
def cmd_build_features_draft(
    matches_path: Annotated[Path, typer.Option(help="Path to matches.parquet")] = Path(
        "data/processed/matches.parquet"
    ),
    picks_path: Annotated[Path, typer.Option(help="Path to match_picks.parquet")] = Path(
        "data/processed/match_picks.parquet"
    ),
    hero_wr_path: Annotated[
        Path, typer.Option(help="Path to hero_patch_winrates.parquet (from scripts/build_meta.py)")
    ] = Path("data/processed/hero_patch_winrates.parquet"),
    out_path: Annotated[Path, typer.Option(help="Output path for post-draft features")] = Path(
        "data/processed/features_draft.parquet"
    ),
) -> None:
    """Build post-draft (H1) features and save to data/processed/features_draft.parquet."""
    matches = pd.read_parquet(matches_path)
    picks = pd.read_parquet(picks_path)
    hero_wr = pd.read_parquet(hero_wr_path)

    feats_draft = build_postdraft_features(matches, picks, hero_wr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats_draft.to_parquet(out_path, index=False)
    typer.echo(f"[OK] Wrote post-draft features ({len(feats_draft)}) rows to {out_path}")


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


# ----------------------------------------------------------------------
# Train (features-aware)
# ----------------------------------------------------------------------
@app.command(name="train", help="Train baseline model on a features parquet and print quick (in-sample) metrics.")
def train_cmd(
    features: Annotated[
        Path,
        typer.Option(help="Path to features parquet (e.g., features_pre.parquet or features_draft.parquet)"),
    ] = Path("data/processed/features_pre.parquet"),
    artifacts_dir: Annotated[Path, typer.Option(help="Where to store models")] = Path("artifacts"),
) -> None:
    """
    Expects columns: 'rating_diff', 'bo' and a target column ('target' or 'team_a_win').
    """
    from sklearn.metrics import brier_score_loss, log_loss

    df = pd.read_parquet(features)

    # Normalize target column name expected by training utilities
    if "team_a_win" not in df.columns and "target" in df.columns:
        df = df.copy()
        df["team_a_win"] = df["target"]
    if "team_a_win" not in df.columns and "target" not in df.columns:
        raise typer.BadParameter("Features must contain a 'target' column (or 'team_a_win').")

    out = train_baseline(df, artifacts_dir=artifacts_dir)
    typer.echo(f"Model saved to: {out}")

    # Quick in-sample metrics (smoke check)
    used_cols = [
        c
        for c in [
            "rating_diff",
            "bo",
            "draft_wr_sum_radiant",
            "draft_wr_avg_radiant",
            "draft_wr_sum_dire",
            "draft_wr_avg_dire",
        ]
        if c in df.columns
    ] or ["rating_diff", "bo"]

    X_eval = df[used_cols].to_numpy(dtype=float)
    y_col = "target" if "target" in df.columns else "team_a_win"
    y = df[y_col].to_numpy(dtype=int)

    model = joblib.load(out)
    y_prob = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "log_loss": float(log_loss(y, y_prob)),
        "brier": float(brier_score_loss(y, y_prob)),
        "features_used": used_cols,
    }
    typer.echo(json.dumps(metrics, indent=2))


# ----------------------------------------------------------------------
# Evaluate (features-aware)
# ----------------------------------------------------------------------
@app.command(name="evaluate", help="Evaluate a saved model on a features parquet.")
def evaluate_cmd(
    features: Annotated[
        Path,
        typer.Option(help="Path to features parquet (e.g., features_pre.parquet or features_draft.parquet)"),
    ] = Path("data/processed/features_pre.parquet"),
    model_path: Annotated[Path, typer.Option(help="Model to evaluate")] = Path("artifacts/baseline_logit.pkl"),
) -> None:
    from sklearn.metrics import brier_score_loss, log_loss

    df = pd.read_parquet(features)

    y_col = "target" if "target" in df.columns else ("team_a_win" if "team_a_win" in df.columns else None)
    if y_col is None:
        raise typer.BadParameter("Features must contain a 'target' column (or 'team_a_win').")
    y = df[y_col].to_numpy(dtype=int)

    used_cols = [
        c
        for c in [
            "rating_diff",
            "bo",
            "draft_wr_sum_radiant",
            "draft_wr_avg_radiant",
            "draft_wr_sum_dire",
            "draft_wr_avg_dire",
        ]
        if c in df.columns
    ] or ["rating_diff", "bo"]
    X_eval = df[used_cols].to_numpy(dtype=float)

    model = joblib.load(model_path)
    y_prob = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "log_loss": float(log_loss(y, y_prob)),
        "brier": float(brier_score_loss(y, y_prob)),
        "features_used": used_cols,
    }
    typer.echo(json.dumps(metrics, indent=2))


# ----------------------------------------------------------------------
# Calibrate (features-aware)
# ----------------------------------------------------------------------
@app.command(name="calibrate", help="Fit a probability calibrator (Platt or Isotonic) on a features parquet.")
def calibrate_cmd(
    method: Annotated[
        str,
        typer.Option(help="Calibration method. One of: 'platt' or 'isotonic' (case-insensitive)."),
    ] = "platt",
    features: Annotated[
        Path,
        typer.Option(help="Path to features parquet (must include 'start_time', 'rating_diff', 'bo', and target)"),
    ] = Path("data/processed/features_pre.parquet"),
    out_model: Annotated[Path, typer.Option(help="Output calibrated model path")] = Path("artifacts/calibrated.pkl"),
    out_plot: Annotated[Path, typer.Option(help="Calibration plot path")] = Path("artifacts/calibration.png"),
    out_metrics: Annotated[Path, typer.Option(help="Calibration metrics JSON path")] = Path(
        "artifacts/calibration_metrics.json"
    ),
) -> None:
    m = method.strip().lower()
    if m not in {"platt", "isotonic"}:
        raise typer.BadParameter("method must be 'platt' or 'isotonic'")

    df = pd.read_parquet(features)

    # Align target name for calibration helpers (expect 'team_a_win')
    if "team_a_win" not in df.columns and "target" in df.columns:
        df = df.rename(columns={"target": "team_a_win"})

    required = {"start_time", "rating_diff", "bo", "team_a_win"}
    missing = required - set(df.columns)
    if missing:
        raise typer.BadParameter(f"Features parquet missing required columns: {sorted(missing)}")

    metrics = calibrate_and_save(
        df=df,
        method=m,
        out_model=str(out_model),
        out_plot=str(out_plot),
        out_metrics=str(out_metrics),
    )
    typer.echo(json.dumps(metrics, indent=2))


# ----------------------------------------------------------------------
# Backtest (rolling / patch-aligned)
# ----------------------------------------------------------------------
@app.command(name="backtest", help="Rolling backtest over time-sorted data (row-based or patch-aligned).")
def backtest_cmd(
    data_path: Annotated[Path, typer.Option(help="Processed Parquet path")] = Path("data/processed/matches.parquet"),
    n_splits: Annotated[int, typer.Option(help="Number of folds")] = 5,
    min_train: Annotated[int, typer.Option(help="Min rows in first train window")] = 800,
    step: Annotated[int, typer.Option(help="Rows to advance per fold (row-based mode)")] = 200,
    out_plot: Annotated[Path, typer.Option(help="Plot output path")] = Path("artifacts/backtest_metrics.png"),
    out_csv: Annotated[Path, typer.Option(help="CSV output path")] = Path("artifacts/backtest_metrics.csv"),
    datetime_col: Annotated[str, typer.Option(help="Timestamp column")] = "start_time",
    snap_to_patch: Annotated[bool, typer.Option(help="If true, test on whole next patch blocks")] = False,
    patch_col: Annotated[str, typer.Option(help="Patch column name (when snap_to_patch=True)")] = "patch",
    embargo_days: Annotated[int, typer.Option(help="Skip first N days of each test patch")] = 0,
) -> None:
    # NOTE: Use the implementation from evaluate/backtest.py
    from .evaluate.backtest import rolling_backtest

    df = pd.read_parquet(data_path)
    out = rolling_backtest(
        df=df,
        datetime_col=datetime_col,
        n_splits=n_splits,
        min_train=min_train,
        step=step,
        out_csv=str(out_csv),
        out_plot=str(out_plot),
        snap_to_patch=snap_to_patch,
        patch_col=patch_col,
        embargo_days=embargo_days,
    )
    typer.echo(f"[backtest] folds produced: {len(out)}")
    typer.echo(f"[backtest] wrote CSV → {out_csv}")
    typer.echo(f"[backtest] wrote PNG → {out_plot}")
    typer.echo(out.to_json(orient="records", indent=2))


# If you intentionally keep a local copy of rolling_backtest in this file,
# make sure the typing matches the FoldResult TypedDict.
from esports_quant.evaluate.backtest import _score_fold, FoldResult  # typed helpers


def rolling_backtest(
    df: pd.DataFrame,
    datetime_col: str,
    n_splits: int = 5,
    min_train: int = 2000,
    step: int = 500,
    out_csv: str | Path | None = None,
    out_plot: str | Path | None = None,
    snap_to_patch: bool = False,
    patch_col: str = "patch",
    embargo_days: int = 0,
) -> pd.DataFrame:
    # Ensure chronological order
    df = df.sort_values(datetime_col).reset_index(drop=True)

    if len(df) < (min_train + step):
        raise ValueError("Not enough rows for requested min_train + step")

    # ✅ Match the TypedDict return type from _score_fold
    results: List[FoldResult] = []
    fold = 0
    i = min_train
    while i + step <= len(df) and fold < n_splits:
        train_df = df.iloc[:i].copy()
        test_df = df.iloc[i : i + step].copy()
        res = _score_fold(train_df, test_df, fold=fold)  # FoldResult
        results.append(res)
        fold += 1
        i += step

    out = pd.DataFrame(results)

    # --- CSV output ---
    if out_csv is not None:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)

    # --- Plot output ---
    if out_plot is not None:
        import matplotlib.pyplot as plt  # lazy import to avoid top-level dep

        p = Path(out_plot)
        p.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        if out.empty:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No folds produced.\nCheck min_train/step/n_splits vs dataset size.",
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
            )
        else:
            ax.plot(out["fold"], out["log_loss"], marker="o", label="log_loss")
            ax.plot(out["fold"], out["brier"], marker="o", label="brier")
            ax.plot(out["fold"], out["ece"], marker="o", label="ece")
            ax.set_xlabel("Fold")
            ax.set_ylabel("Score (lower is better)")
            ax.set_title("Rolling backtest metrics")
            ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)

    return out
