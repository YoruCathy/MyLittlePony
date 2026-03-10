#!/usr/bin/env python
"""Gait embedding visualization for riding sessions.

Loads two sessions (walk / walk+trot), extracts windowed features from
both the computed metrics and raw IMU data, and produces PCA + t-SNE
scatter plots to see whether walk and trot separate in feature space.

Usage
-----
    python scripts/visualize_gait_embeddings.py

Outputs are saved to ``outputs/embeddings/``.
"""

from pathlib import Path

from mlp.data.loader import load_session
from mlp.analysis.features import (
    extract_metric_windows,
    extract_imu_windows,
    concatenate_features,
)
from mlp.visualization.embeddings import plot_embeddings, plot_temporal_embedding

# ── paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUT_DIR = Path("outputs/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_1 = DATA_DIR / "session_export_2026-03-08T13-53-06"  # walk only
SESSION_2 = DATA_DIR / "session_export_2026-03-08T14-27-09"  # walk + trot

# ── parameters ────────────────────────────────────────────────────────────
WINDOW_MS = 2000   # 2-second windows
STRIDE_MS = 500    # 0.5-second stride (75% overlap)
PERPLEXITY = 40


def main() -> None:
    print("Loading sessions …")
    s1 = load_session(SESSION_1)
    s2 = load_session(SESSION_2)

    # ── 1. Computed-metrics embeddings ────────────────────────────────────
    print("Extracting metric-window features …")
    wf1_m = extract_metric_windows(
        s1.metrics, WINDOW_MS, STRIDE_MS,
        label="walk", session_id="session-1 (walk)",
    )
    wf2_m = extract_metric_windows(
        s2.metrics, WINDOW_MS, STRIDE_MS,
        label="walk+trot", session_id="session-2 (walk+trot)",
    )
    wf_metrics = concatenate_features(wf1_m, wf2_m)
    print(f"  → {wf_metrics.features.shape[0]} windows, "
          f"{wf_metrics.features.shape[1]} features")

    print("Computing metric embeddings …")
    plot_embeddings(
        wf_metrics,
        title_prefix="Metrics — ",
        perplexity=PERPLEXITY,
        save_path=OUT_DIR / "metrics_embeddings.png",
    )
    print(f"  → saved {OUT_DIR / 'metrics_embeddings.png'}")

    plot_temporal_embedding(
        wf_metrics,
        title_prefix="Metrics — ",
        perplexity=PERPLEXITY,
        save_path=OUT_DIR / "metrics_temporal.png",
    )
    print(f"  → saved {OUT_DIR / 'metrics_temporal.png'}")

    # ── 2. Raw-IMU embeddings ─────────────────────────────────────────────
    print("Extracting raw-IMU-window features …")
    wf1_i = extract_imu_windows(
        s1.raw_imu, WINDOW_MS, STRIDE_MS,
        label="walk", session_id="session-1 (walk)",
    )
    wf2_i = extract_imu_windows(
        s2.raw_imu, WINDOW_MS, STRIDE_MS,
        label="walk+trot", session_id="session-2 (walk+trot)",
    )
    wf_imu = concatenate_features(wf1_i, wf2_i)
    print(f"  → {wf_imu.features.shape[0]} windows, "
          f"{wf_imu.features.shape[1]} features")

    print("Computing IMU embeddings …")
    plot_embeddings(
        wf_imu,
        title_prefix="Raw IMU — ",
        perplexity=PERPLEXITY,
        save_path=OUT_DIR / "imu_embeddings.png",
    )
    print(f"  → saved {OUT_DIR / 'imu_embeddings.png'}")

    plot_temporal_embedding(
        wf_imu,
        title_prefix="Raw IMU — ",
        perplexity=PERPLEXITY,
        save_path=OUT_DIR / "imu_temporal.png",
    )
    print(f"  → saved {OUT_DIR / 'imu_temporal.png'}")

    print("\nDone! Check outputs in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
