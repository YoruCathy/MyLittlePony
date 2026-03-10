"""Common type aliases used across the library."""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
"""Accepts either a string or a :class:`~pathlib.Path`."""
