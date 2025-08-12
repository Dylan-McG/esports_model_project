# Purpose: Explicit package marker (keeps tools happy).
from __future__ import annotations  # future typing

from typing import List  # for precise __all__ annotation

__all__: List[str] = []  # no public API exports yet
