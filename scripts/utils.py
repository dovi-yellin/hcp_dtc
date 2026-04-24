"""Compatibility wrapper for legacy scripts importing filtering helpers from ``utils``."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from infrastructure.filters import (
    butter_bandpass,
    butter_bandpass_filter,
    butter_filter,
    butter_highpass_filter,
    butter_lowpass_filter,
)

__all__ = [
    "butter_bandpass",
    "butter_bandpass_filter",
    "butter_filter",
    "butter_highpass_filter",
    "butter_lowpass_filter",
]
