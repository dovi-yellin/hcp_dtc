"""Compatibility wrapper for legacy scripts importing ``initParams``."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from infrastructure.init_params import initParams, read_config

__all__ = ["initParams", "read_config"]
