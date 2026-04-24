"""Compatibility wrapper for legacy scripts importing ``models.RateModel``.

The canonical implementation lives in ``infrastructure.rate_model``.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from infrastructure.rate_model import RateModel

__all__ = ["RateModel"]
