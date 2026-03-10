#!/usr/bin/env python
"""Walk gait prediction pipeline.

1. Train a novelty detector on session 1 (known walk).
2. Predict walk vs. trot on session 2 (walk + trot).
3. Re-train a supervised classifier using the pseudo-labels.
4. Produce timeline + feature-importance visualizations.

Usage
-----
    python scripts/predict_walk.py

Outputs are saved to ``outputs/predictions/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mlp.data.loader import load_session
from mlp.analysis.features import (
    extract_metric_windows,
    concatenate_features,
)
from mlp.analysis.gait_predictor import (
    WalkClassifier,
    WalkNoveltyDetector,
)

# ── paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUT_DIR = Path("outputs/predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_1 = DATA_DIR / "session_export_2026-03-08T13-53-06"  # walk only
SESSION_2 = DATA_DIR / "session_export_2026-03-08T14-27-09"  # walk + trot

# ── parameters ────────────────────────────────────────────────────────────
WINDOW_MS = 2000
STRIDE_MS = 500


def main() -> None:
    print("Loading sessions …")
    s1 = load_session(SESSION_1)
    s2 = load_session(SESSION_2)

    # ── Extract windowed features ─────────────────────────────────────────
    print("Extracting windowed features …")
    wf_walk = extract_metric_windows(
        s1.metrics, WINDOW_MS, STRIDE_MS,
        label="walk", session_id="session-1",
    )
    wf_test = extract_metric_windows(
        s2.metrics, WINDOW_MS, STRIDE_MS,
        label="unknown", session_id="session-2",
    )
    print(f"  walk train: {wf_walk.features.shape[0]} windows")
    print(f"  session-2:  {wf_test.features.shape[0]} windows")

    # ── 1. Novelty detection ─────────────────────────────────────────────
    print("\n── Novelty detector (Isolation Forest) ──")
    detector = WalkNoveltyDetector(method="iforest", contamination=0.05)
    detector.fit(wf_walk)

    nd_result = detector.predict(wf_test)
    nd_summary = nd_result.summary()
    print(f"  Walk windows:  {nd_summary['walk_windows']}  "
          f"({nd_summary['walk_pct']:.1f}%)")
    print(f"  Trot windows:  {nd_summary['trot_windows']}  "
          f"({100 - nd_summary['walk_pct']:.1f}%)")

    # ── 2. Supervised classifier using pseudo-labels ─────────────────────
    print("\n── Supervised classifier (Random Forest) ──")
    # Build training set: session-1 (all walk) + session-2 pseudo-labelled
    wf_test_labelled = extract_metric_windows(
        s2.metrics, WINDOW_MS, STRIDE_MS,
        label=None, session_id="session-2",
    )
    wf_test_labelled.labels = nd_result.predictions  # pseudo-labels

    wf_train = concatenate_features(wf_walk, wf_test_labelled)
    clf = WalkClassifier(n_estimators=300)
    clf.fit(wf_train)

    clf_result = clf.predict(wf_test)
    clf_summary = clf_result.summary()
    print(f"  Walk windows:  {clf_summary['walk_windows']}  "
          f"({clf_summary['walk_pct']:.1f}%)")
    print(f"  Trot windows:  {clf_summary['trot_windows']}  "
          f"({100 - clf_summary['walk_pct']:.1f}%)")

    # ── 3. Visualizations ────────────────────────────────────────────────
    print("\nGenerating plots …")

    # (a) Timeline: novelty score + predictions for session 2
    _plot_timeline(nd_result, clf_result, s2.info.duration_ms, OUT_DIR)

    # (b) Feature importance (top 20)
    _plot_feature_importance(clf, wf_train.feature_names, OUT_DIR)

    # (c) Score distribution
    _plot_score_distributions(nd_result, clf_result, OUT_DIR)

    # (d) Metric comparison: walk vs trot windows
    _plot_metric_comparison(wf_test, clf_result, OUT_DIR)

    print(f"\nDone! Outputs in: {OUT_DIR.resolve()}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_timeline(
    nd_result, clf_result, duration_ms: int, out_dir: Path,
) -> None:
    """Timeline of walk/trot predictions over session 2."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

    t_sec = nd_result.timestamps_ms / 1000

    # (1) Novelty score
    ax = axes[0]
    ax.plot(t_sec, nd_result.scores, linewidth=0.5, color="steelblue", alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="decision boundary")
    ax.fill_between(t_sec, nd_result.scores, 0,
                     where=nd_result.scores < 0, alpha=0.3, color="orange",
                     label="anomaly (trot-like)")
    ax.set_ylabel("Novelty score")
    ax.set_title("Isolation Forest — novelty score over time (session 2)")
    ax.legend(loc="upper right", fontsize=8)

    # (2) Novelty binary prediction
    ax = axes[1]
    walk_mask_nd = nd_result.walk_mask
    ax.fill_between(t_sec, 0, 1, where=walk_mask_nd,
                     color="tab:blue", alpha=0.4, label="walk")
    ax.fill_between(t_sec, 0, 1, where=~walk_mask_nd,
                     color="tab:orange", alpha=0.4, label="trot")
    ax.set_ylabel("Gait (novelty)")
    ax.set_title("Novelty detector — predicted gait")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)

    # (3) Classifier prediction
    ax = axes[2]
    walk_mask_clf = clf_result.walk_mask
    ax.fill_between(t_sec, 0, 1, where=walk_mask_clf,
                     color="tab:blue", alpha=0.4, label="walk")
    ax.fill_between(t_sec, 0, 1, where=~walk_mask_clf,
                     color="tab:orange", alpha=0.4, label="trot")
    ax.set_ylabel("Gait (classifier)")
    ax.set_title("Random Forest — predicted gait")
    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path = out_dir / "timeline_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def _plot_feature_importance(clf: WalkClassifier, names: list[str], out_dir: Path) -> None:
    """Bar chart of top feature importances."""
    top = clf.top_features(names, n=20)
    feat_names = [t[0] for t in top][::-1]
    importances = [t[1] for t in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(feat_names)), importances, color="steelblue")
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_xlabel("Gini importance")
    ax.set_title("Top 20 features for walk vs. trot classification")
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def _plot_score_distributions(nd_result, clf_result, out_dir: Path) -> None:
    """Histograms of walk vs. trot scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Novelty scores
    ax = axes[0]
    walk_s = nd_result.scores[nd_result.walk_mask]
    trot_s = nd_result.scores[nd_result.trot_mask]
    ax.hist(walk_s, bins=60, alpha=0.6, color="tab:blue", label="walk", density=True)
    ax.hist(trot_s, bins=60, alpha=0.6, color="tab:orange", label="trot", density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Novelty score")
    ax.set_ylabel("Density")
    ax.set_title("Novelty detector — score distribution")
    ax.legend()

    # Classifier walk probability
    ax = axes[1]
    walk_p = clf_result.scores[clf_result.walk_mask]
    trot_p = clf_result.scores[clf_result.trot_mask]
    ax.hist(walk_p, bins=60, alpha=0.6, color="tab:blue", label="walk", density=True)
    ax.hist(trot_p, bins=60, alpha=0.6, color="tab:orange", label="trot", density=True)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("P(walk)")
    ax.set_ylabel("Density")
    ax.set_title("Classifier — walk probability distribution")
    ax.legend()

    fig.tight_layout()
    path = out_dir / "score_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


def _plot_metric_comparison(wf: "WindowedFeatures", result, out_dir: Path) -> None:
    """Box-plot comparison of key metrics for walk vs. trot windows."""
    # Pick angular velocity and linear acceleration features (mean stat)
    mean_cols = [i for i, n in enumerate(wf.feature_names) if n.endswith(".mean")]
    mean_names = [wf.feature_names[i] for i in mean_cols]

    walk_mask = result.walk_mask
    trot_mask = result.trot_mask

    n_metrics = len(mean_cols)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (col_i, name) in enumerate(zip(mean_cols, mean_names)):
        if idx >= len(axes):
            break
        ax = axes[idx]
        walk_vals = wf.features[walk_mask, col_i]
        trot_vals = wf.features[trot_mask, col_i]
        # Filter finite
        walk_vals = walk_vals[np.isfinite(walk_vals)]
        trot_vals = trot_vals[np.isfinite(trot_vals)]
        bp = ax.boxplot(
            [walk_vals, trot_vals],
            tick_labels=["walk", "trot"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("tab:blue")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("tab:orange")
        bp["boxes"][1].set_alpha(0.5)
        ax.set_title(name.replace(".mean", ""), fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for idx in range(len(mean_cols), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Metric distributions: walk vs. trot (classifier labels)", fontsize=12)
    fig.tight_layout()
    path = out_dir / "metric_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


if __name__ == "__main__":
    main()
