"""Data-class representation of a riding session."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Supporting dataclasses for structured session_info.json content
# ---------------------------------------------------------------------------

@dataclass
class StickyConfig:
    """A single sensor (sticky) placement in the task."""

    segment: str
    placement: str | None
    type: str  # "SENSING" or "FEEDBACK"


@dataclass
class MetricConfig:
    """Configuration for a computed metric."""

    type: str  # e.g. "SEGMENT"
    target_id: str
    min_threshold: float | None
    max_threshold: float | None
    include_velocity: bool
    included_sub_metrics: list[str]


@dataclass
class CalibrationPose:
    """A single calibration pose definition."""

    name: str
    instruction: str
    duration_seconds: int


@dataclass
class SensorCalibration:
    """Calibration data for one sensor in a pose."""

    orientation: dict[str, float]  # quaternion {w, x, y, z}
    gravity_in_sensor: dict[str, float]  # {x, y, z}
    skin_normal: dict[str, float]  # {x, y, z}


@dataclass
class DeviceInfo:
    """Mapping between a sticky id and its physical device."""

    device_address: str
    device_id: str
    assigned_sticky_id: str
    type: str  # "SENSING" or "FEEDBACK"


@dataclass
class PolicyTarget:
    """A target defined in the feedback policy."""

    name: str
    metric: str
    default_value: float
    default_tolerance: float
    value_min: float
    value_max: float
    tolerance_min: float
    tolerance_max: float


@dataclass
class SessionInfo:
    """Parsed metadata from session_info.json."""

    # App / device info
    app_version: str
    device_model: str
    device_manufacturer: str

    # Timing
    timestamp_ms: int
    start_time_ms: int
    end_time_ms: int
    duration_ms: int

    # Task definition
    task_name: str
    task_description: str
    stickies: list[StickyConfig]
    metric_names: list[str]
    metric_configs: list[MetricConfig]
    calibration_poses: list[CalibrationPose]

    # Policy
    policy_name: str
    policy_targets: list[PolicyTarget]

    # Device mapping
    devices: dict[str, DeviceInfo]

    # Calibration data (per pose → per sensor)
    pose_calibrations: dict[str, dict[str, SensorCalibration]]

    # Parameter values (e.g. target overrides stored at session end)
    parameter_values: dict[str, float]

    # Keep the raw dict around for anything we haven't parsed yet
    raw: dict[str, Any] = field(repr=False)

    # ----- computed helpers ------------------------------------------------

    @property
    def start_datetime(self) -> datetime:
        """Session start as a UTC datetime."""
        return datetime.fromtimestamp(self.start_time_ms / 1000, tz=timezone.utc)

    @property
    def end_datetime(self) -> datetime:
        """Session end as a UTC datetime."""
        return datetime.fromtimestamp(self.end_time_ms / 1000, tz=timezone.utc)

    @property
    def duration(self) -> timedelta:
        """Session duration as a timedelta."""
        return timedelta(milliseconds=self.duration_ms)

    @property
    def sensor_segments(self) -> list[str]:
        """List of sensor segment ids (sensing stickies only)."""
        return [s.segment for s in self.stickies if s.type == "SENSING"]


# ---------------------------------------------------------------------------
# Main session container
# ---------------------------------------------------------------------------

@dataclass
class RidingSession:
    """Container for all data from a single riding session export.

    Attributes
    ----------
    path : Path
        Directory the session was loaded from.
    info : SessionInfo
        Parsed session metadata.
    raw_imu : pd.DataFrame
        Raw IMU readings (timestamp_ms, sticky_id, ax, ay, az, wx, wy, wz).
    metrics : pd.DataFrame
        Computed metrics time-series.
    feedback_log : pd.DataFrame
        Feedback events log.
    """

    path: Path
    info: SessionInfo
    raw_imu: pd.DataFrame
    metrics: pd.DataFrame
    feedback_log: pd.DataFrame

    # ----- convenience accessors -------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable session name derived from its directory."""
        return self.path.name

    @property
    def sensor_segments(self) -> list[str]:
        """List of sensor body-segment ids in this session."""
        return self.info.sensor_segments

    def imu_for_segment(self, segment: str) -> pd.DataFrame:
        """Return raw IMU rows for a given sensor segment.

        Parameters
        ----------
        segment : str
            Segment id, e.g. ``"hand_L"``, ``"head"``.

        Returns
        -------
        pd.DataFrame
            Filtered raw IMU data for that segment.
        """
        return self.raw_imu[self.raw_imu["sticky_id"] == segment].copy()

    def metric_columns_for_segment(self, segment: str) -> list[str]:
        """Return the metric column names that belong to *segment*."""
        prefix = f"{segment}."
        return [c for c in self.metrics.columns if c.startswith(prefix)]

    def metrics_for_segment(self, segment: str) -> pd.DataFrame:
        """Return the metric columns for *segment* together with timestamp."""
        cols = ["timestamp_ms"] + self.metric_columns_for_segment(segment)
        return self.metrics[cols].copy()

    def duration_seconds(self) -> float:
        """Session duration in seconds."""
        return self.info.duration_ms / 1000.0

    def __repr__(self) -> str:
        n_imu = len(self.raw_imu)
        n_met = len(self.metrics)
        dur = f"{self.duration_seconds():.1f}s"
        return (
            f"RidingSession({self.name!r}, duration={dur}, "
            f"imu_rows={n_imu}, metric_rows={n_met})"
        )
