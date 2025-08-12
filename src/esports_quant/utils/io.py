# Purpose: Lightweight IO utilities for directories and pickle saves/loads.

from __future__ import annotations  # ensure future typing behavior is active first

import pickle  # serialize/deserialize Python objects
from pathlib import Path  # path handling across OSes
from typing import Any  # for flexible return/param typing


def ensure_dir(path: str | Path) -> Path:
    # Normalize to Path so we can call mkdir and return a Path consistently
    p = Path(path)
    # Create the directory tree if it doesn't exist
    p.mkdir(parents=True, exist_ok=True)
    # Return the Path so callers can chain operations
    return p


def save_pickle(obj: Any, path: str | Path) -> None:
    # Normalize the output path
    path = Path(path)
    # Ensure the parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open the file in binary write mode and dump the object with the highest protocol
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | Path) -> Any:
    # Open the file in binary read mode and load the object back
    with open(path, "rb") as f:
        return pickle.load(f)
