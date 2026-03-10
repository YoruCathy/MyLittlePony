"""MyLittlePony (mlp) — Riding data analysis library."""

__version__ = "0.1.0"

from mlp.data.loader import load_session, load_sessions
from mlp.data.session import RidingSession

__all__ = [
    "load_session",
    "load_sessions",
    "RidingSession",
]
