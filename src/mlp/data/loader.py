"""Load riding-session exports from disk.

A session export is a directory that contains:

- ``session_info.json`` — session metadata, task config, calibration, policy
- ``raw_imu.csv`` — raw accelerometer + gyroscope readings per sensor
- ``metrics.csv`` — computed metric time-series
- ``feedback_log.csv`` — feedback-rule activations
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mlp.data.session import (
    CalibrationPose,
    DeviceInfo,
    MetricConfig,
    PolicyTarget,
    RidingSession,
    SensorCalibration,
    SessionInfo,
    StickyConfig,
)

logger = logging.getLogger(__name__)

# Names of the expected files inside a session export directory
_SESSION_INFO_FILE = "session_info.json"
_RAW_IMU_FILE = "raw_imu.csv"
_METRICS_FILE = "metrics.csv"
_FEEDBACK_LOG_FILE = "feedback_log.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_session(session_dir: str | Path) -> RidingSession:
    """Load a single session export directory.

    Parameters
    ----------
    session_dir : str | Path
        Path to a ``session_export_*`` directory.

    Returns
    -------
    RidingSession
        Fully loaded session object.

    Raises
    ------
    FileNotFoundError
        If the directory or any required file is missing.
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    info = _load_session_info(session_dir / _SESSION_INFO_FILE)
    raw_imu = _load_csv(session_dir / _RAW_IMU_FILE)
    metrics = _load_csv(session_dir / _METRICS_FILE)
    feedback_log = _load_csv(session_dir / _FEEDBACK_LOG_FILE)

    logger.info(
        "Loaded session %s  —  %d IMU rows, %d metric rows, duration %.1fs",
        session_dir.name,
        len(raw_imu),
        len(metrics),
        info.duration_ms / 1000,
    )

    return RidingSession(
        path=session_dir,
        info=info,
        raw_imu=raw_imu,
        metrics=metrics,
        feedback_log=feedback_log,
    )


def load_sessions(data_dir: str | Path) -> list[RidingSession]:
    """Load every session export found under *data_dir*.

    Directories are discovered by globbing for ``session_export_*/session_info.json``.

    Parameters
    ----------
    data_dir : str | Path
        Parent directory that contains one or more ``session_export_*`` folders.

    Returns
    -------
    list[RidingSession]
        Sorted by session start time (ascending).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    session_dirs = sorted(
        p.parent for p in data_dir.glob(f"session_export_*/{_SESSION_INFO_FILE}")
    )

    if not session_dirs:
        logger.warning("No session exports found under %s", data_dir)
        return []

    sessions = [load_session(d) for d in session_dirs]
    sessions.sort(key=lambda s: s.info.start_time_ms)
    return sessions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    if not path.is_file():
        raise FileNotFoundError(f"Expected CSV file not found: {path}")
    return pd.read_csv(path)


def _load_session_info(path: Path) -> SessionInfo:
    """Parse ``session_info.json`` into a :class:`SessionInfo`."""
    if not path.is_file():
        raise FileNotFoundError(f"Session info file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    task: dict[str, Any] = raw.get("task", {})
    policy: dict[str, Any] = raw.get("policy", {})
    device_map: dict[str, Any] = raw.get("deviceMapping", {}).get(
        "devicesByStickyId", {}
    )
    cal_data: dict[str, Any] = raw.get("calibrationData", {})

    return SessionInfo(
        app_version=raw.get("appVersion", ""),
        device_model=raw.get("deviceModel", ""),
        device_manufacturer=raw.get("deviceManufacturer", ""),
        timestamp_ms=raw.get("timestampMs", 0),
        start_time_ms=raw.get("startTimeMs", 0),
        end_time_ms=raw.get("endTimeMs", 0),
        duration_ms=raw.get("durationMs", 0),
        task_name=task.get("name", ""),
        task_description=task.get("description", ""),
        stickies=_parse_stickies(task.get("stickies", [])),
        metric_names=task.get("metrics", []),
        metric_configs=_parse_metric_configs(task.get("metricConfigs", [])),
        calibration_poses=_parse_calibration_poses(
            task.get("calibration", {}).get("poses", [])
        ),
        policy_name=policy.get("name", ""),
        policy_targets=_parse_policy_targets(policy.get("targets", [])),
        devices=_parse_devices(device_map),
        pose_calibrations=_parse_pose_calibrations(
            cal_data.get("poseCalibrations", {})
        ),
        parameter_values=raw.get("parameterValues", {}),
        raw=raw,
    )


# -- tiny parsers for nested structures ------------------------------------


def _parse_stickies(items: list[dict]) -> list[StickyConfig]:
    return [
        StickyConfig(
            segment=s["segment"],
            placement=s.get("placement"),
            type=s.get("type", "SENSING"),
        )
        for s in items
    ]


def _parse_metric_configs(items: list[dict]) -> list[MetricConfig]:
    return [
        MetricConfig(
            type=m.get("type", ""),
            target_id=m.get("targetId", ""),
            min_threshold=m.get("minThreshold"),
            max_threshold=m.get("maxThreshold"),
            include_velocity=m.get("includeVelocity", False),
            included_sub_metrics=m.get("includedSubMetrics", []),
        )
        for m in items
    ]


def _parse_calibration_poses(items: list[dict]) -> list[CalibrationPose]:
    return [
        CalibrationPose(
            name=p.get("name", ""),
            instruction=p.get("instruction", ""),
            duration_seconds=p.get("durationSeconds", 0),
        )
        for p in items
    ]


def _parse_policy_targets(items: list[dict]) -> list[PolicyTarget]:
    return [
        PolicyTarget(
            name=t.get("name", ""),
            metric=t.get("metric", ""),
            default_value=t.get("defaultValue", 0.0),
            default_tolerance=t.get("defaultTolerance", 0.0),
            value_min=t.get("valueMin", 0.0),
            value_max=t.get("valueMax", 0.0),
            tolerance_min=t.get("toleranceMin", 0.0),
            tolerance_max=t.get("toleranceMax", 0.0),
        )
        for t in items
    ]


def _parse_devices(device_map: dict[str, dict]) -> dict[str, DeviceInfo]:
    return {
        sticky_id: DeviceInfo(
            device_address=d.get("deviceAddress", ""),
            device_id=d.get("deviceId", ""),
            assigned_sticky_id=d.get("assignedStickyId", sticky_id),
            type=d.get("type", ""),
        )
        for sticky_id, d in device_map.items()
    }


def _parse_pose_calibrations(
    pose_cals: dict[str, dict],
) -> dict[str, dict[str, SensorCalibration]]:
    result: dict[str, dict[str, SensorCalibration]] = {}
    for pose_name, sensors in pose_cals.items():
        result[pose_name] = {}
        for sensor_id, cal in sensors.items():
            result[pose_name][sensor_id] = SensorCalibration(
                orientation=cal.get("orientation", {}),
                gravity_in_sensor=cal.get("gravityInSensor", {}),
                skin_normal=cal.get("skinNormal", {}),
            )
    return result
