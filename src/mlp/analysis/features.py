"""Windowed feature extraction from riding session data.

Extracts statistical features from sliding windows over the metrics or
raw IMU time-series, producing a feature matrix suitable for
dimensionality reduction / clustering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowedFeatures:
    """Result of windowed feature extraction.

    Attributes
    ----------
    features : np.ndarray
        Feature matrix of shape ``(n_windows, n_features)``.
    feature_names : list[str]
        Human-readable name for each feature column.
    timestamps_ms : np.ndarray
        Centre timestamp (ms) of each window.
    labels : np.ndarray | None
        Optional per-window label array (e.g. gait type).
    session_ids : np.ndarray | None
        Optional per-window session identifier.
    """

    features: np.ndarray
    feature_names: list[str]
    timestamps_ms: np.ndarray
    labels: np.ndarray | None = None
    session_ids: np.ndarray | None = None


def extract_metric_windows(
    metrics: pd.DataFrame,
    window_size_ms: float = 2000,
    stride_ms: float = 500,
    label: str | None = None,
    session_id: str | None = None,
) -> WindowedFeatures:
    """Slide a window over the *metrics* DataFrame and compute features.

    For every signal column the following statistics are computed per
    window: **mean, std, min, max, range, median, energy (mean of
    squared values)**.

    Parameters
    ----------
    metrics : pd.DataFrame
        Must contain a ``timestamp_ms`` column; all other columns are
        treated as signals.
    window_size_ms : float
        Width of the sliding window in milliseconds.
    stride_ms : float
        Step between consecutive windows in milliseconds.
    label : str | None
        If given, every window receives this label.
    session_id : str | None
        If given, stored in the result for later identification.

    Returns
    -------
    WindowedFeatures
    """
    ts = metrics["timestamp_ms"].values
    signal_cols = [c for c in metrics.columns if c != "timestamp_ms"]
    signals = metrics[signal_cols].values  # (T, C)

    t_start = ts[0]
    t_end = ts[-1]

    window_starts = np.arange(t_start, t_end - window_size_ms + 1, stride_ms)

    feature_rows: list[np.ndarray] = []
    centre_times: list[float] = []

    for ws in window_starts:
        we = ws + window_size_ms
        mask = (ts >= ws) & (ts < we)
        if mask.sum() < 5:  # skip near-empty windows
            continue
        chunk = signals[mask]  # (W, C)

        feats = _compute_window_stats(chunk)
        feature_rows.append(feats)
        centre_times.append(ws + window_size_ms / 2)

    if not feature_rows:
        return WindowedFeatures(
            features=np.empty((0, 0)),
            feature_names=[],
            timestamps_ms=np.empty(0),
        )

    stat_names = ["mean", "std", "min", "max", "range", "median", "energy"]
    feature_names = [
        f"{col}.{stat}" for col in signal_cols for stat in stat_names
    ]

    n_windows = len(feature_rows)
    features = np.vstack(feature_rows)
    timestamps = np.array(centre_times)

    labels_arr = np.full(n_windows, label) if label is not None else None
    session_arr = np.full(n_windows, session_id) if session_id is not None else None

    return WindowedFeatures(
        features=features,
        feature_names=feature_names,
        timestamps_ms=timestamps,
        labels=labels_arr,
        session_ids=session_arr,
    )


def extract_imu_windows(
    raw_imu: pd.DataFrame,
    window_size_ms: float = 2000,
    stride_ms: float = 500,
    label: str | None = None,
    session_id: str | None = None,
) -> WindowedFeatures:
    """Slide a window over *raw_imu* and compute per-segment features.

    Raw IMU has columns ``timestamp_ms, sticky_id, ax, ay, az, wx, wy, wz``.
    We pivot by segment so each window produces statistics across all
    sensor axes for all segments.

    Parameters
    ----------
    raw_imu : pd.DataFrame
        Raw IMU DataFrame with ``sticky_id`` column.
    window_size_ms, stride_ms, label, session_id :
        Same as :func:`extract_metric_windows`.

    Returns
    -------
    WindowedFeatures
    """
    imu_cols = ["ax", "ay", "az", "wx", "wy", "wz"]
    segments = sorted(raw_imu["sticky_id"].unique())

    # Pivot to wide format: one row per timestamp, columns = segment.axis
    pivot_parts: list[pd.DataFrame] = []
    for seg in segments:
        seg_df = raw_imu[raw_imu["sticky_id"] == seg][["timestamp_ms"] + imu_cols].copy()
        seg_df = seg_df.rename(columns={c: f"{seg}.{c}" for c in imu_cols})
        pivot_parts.append(seg_df)

    # Merge all segments on timestamp (outer join keeps all timestamps)
    merged = pivot_parts[0]
    for part in pivot_parts[1:]:
        merged = pd.merge(merged, part, on="timestamp_ms", how="outer")
    merged = merged.sort_values("timestamp_ms").reset_index(drop=True)
    # Forward-fill small gaps so windows work smoothly
    merged = merged.ffill().bfill()

    return extract_metric_windows(
        merged,
        window_size_ms=window_size_ms,
        stride_ms=stride_ms,
        label=label,
        session_id=session_id,
    )


def concatenate_features(*wfs: WindowedFeatures) -> WindowedFeatures:
    """Stack multiple :class:`WindowedFeatures` vertically.

    All inputs must have the same ``feature_names``.
    """
    if not wfs:
        raise ValueError("Need at least one WindowedFeatures")
    names = wfs[0].feature_names
    for wf in wfs[1:]:
        if wf.feature_names != names:
            raise ValueError("Feature names do not match across inputs")

    features = np.vstack([wf.features for wf in wfs])
    timestamps = np.concatenate([wf.timestamps_ms for wf in wfs])

    labels_parts = [wf.labels for wf in wfs]
    labels = np.concatenate(labels_parts) if all(l is not None for l in labels_parts) else None

    sid_parts = [wf.session_ids for wf in wfs]
    session_ids = np.concatenate(sid_parts) if all(s is not None for s in sid_parts) else None

    return WindowedFeatures(
        features=features,
        feature_names=names,
        timestamps_ms=timestamps,
        labels=labels,
        session_ids=session_ids,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _compute_window_stats(chunk: np.ndarray) -> np.ndarray:
    """Compute per-column statistics and flatten into a 1-D feature vector."""
    # chunk: (W, C)
    mean = chunk.mean(axis=0)
    std = chunk.std(axis=0)
    mn = chunk.min(axis=0)
    mx = chunk.max(axis=0)
    rng = mx - mn
    med = np.median(chunk, axis=0)
    energy = (chunk ** 2).mean(axis=0)
    # stack shape: (7, C) → flatten to (7*C,)
    return np.concatenate([mean, std, mn, mx, rng, med, energy])
