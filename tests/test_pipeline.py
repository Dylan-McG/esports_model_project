# Purpose: Fast end-to-end test using the same interpreter as pytest.

from __future__ import annotations  # future-friendly typing

import json  # parse metrics JSON
import subprocess  # spawn CLI
import sys  # current interpreter
from pathlib import Path  # paths


def run_module(module: str, args: list[str]) -> subprocess.CompletedProcess:
    # Build a cmd that uses the SAME interpreter as pytest (works with Poetry venv)
    cmd = [sys.executable, "-m", module, *args]
    # Execute and capture output, raising on non-zero exit
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def test_end_to_end(tmp_path: Path) -> None:
    # Use a temporary directory for isolation
    data_dir = tmp_path / "data" / "processed"
    artifacts_dir = tmp_path / "artifacts"

    # 1) Ingest (dummy)
    run_module("esports_quant.cli", ["ingest", "--out-dir", str(data_dir)])
    assert (data_dir / "matches.parquet").exists()

    # 2) Train
    run_module(
        "esports_quant.cli",
        [
            "train",
            "--data-path",
            str(data_dir / "matches.parquet"),
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )
    assert (artifacts_dir / "baseline_logit.pkl").exists()

    # 3) Evaluate
    cp = run_module(
        "esports_quant.cli",
        ["evaluate", "--artifacts-dir", str(artifacts_dir)],
    )
    metrics = json.loads(cp.stdout.strip())
    assert {"log_loss", "brier", "ece"}.issubset(metrics.keys())
