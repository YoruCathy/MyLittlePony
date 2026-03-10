"""Embedding-based visualization of riding session data.

Produces 2-D scatter plots of windowed features via PCA and t-SNE,
coloured by gait label / session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlp.analysis.features import WindowedFeatures


def _preprocess_features(X_raw: np.ndarray) -> np.ndarray:
    """Impute, drop constant columns, clip, and standardise."""
    X = np.array(X_raw, dtype=np.float64)
    # Replace inf with NaN, then impute
    X[~np.isfinite(X)] = np.nan
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    # Drop zero-variance (constant) columns
    col_std = X.std(axis=0)
    keep = col_std > 1e-12
    X = X[:, keep]
    # Standardise
    X = StandardScaler().fit_transform(X)
    # Clip extremes to prevent overflow in PCA/t-SNE matmuls
    np.clip(X, -10, 10, out=X)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def plot_embeddings(
    wf: WindowedFeatures,
    *,
    title_prefix: str = "",
    perplexity: float = 30,
    figsize: tuple[float, float] = (16, 12),
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Create a 2×2 figure with PCA and t-SNE embeddings.

    Top row is coloured by **label** (gait), bottom row by **session id**.
    Left column = PCA, right column = t-SNE.

    Parameters
    ----------
    wf : WindowedFeatures
        Combined feature windows (should have labels and session_ids set).
    title_prefix : str
        Prepended to subplot titles.
    perplexity : float
        t-SNE perplexity.
    figsize : tuple
        Figure size in inches.
    save_path : str | Path | None
        If given, save figure to this path.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    X = _preprocess_features(wf.features)

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_

    # --- t-SNE ---
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, len(X) // 4)),
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Colour maps
    labels = wf.labels if wf.labels is not None else np.zeros(len(X))
    sessions = wf.session_ids if wf.session_ids is not None else np.zeros(len(X))

    unique_labels = np.unique(labels)
    unique_sessions = np.unique(sessions)

    label_colors = _make_color_map(unique_labels)
    session_colors = _make_color_map(unique_sessions, cmap_name="Set2")

    # Top-left: PCA by label
    _scatter(axes[0, 0], X_pca, labels, label_colors,
             title=f"{title_prefix}PCA — by gait "
                   f"(var: {var_explained[0]:.1%}+{var_explained[1]:.1%})",
             xlabel="PC 1", ylabel="PC 2")

    # Top-right: t-SNE by label
    _scatter(axes[0, 1], X_tsne, labels, label_colors,
             title=f"{title_prefix}t-SNE — by gait",
             xlabel="t-SNE 1", ylabel="t-SNE 2")

    # Bottom-left: PCA by session
    _scatter(axes[1, 0], X_pca, sessions, session_colors,
             title=f"{title_prefix}PCA — by session",
             xlabel="PC 1", ylabel="PC 2")

    # Bottom-right: t-SNE by session
    _scatter(axes[1, 1], X_tsne, sessions, session_colors,
             title=f"{title_prefix}t-SNE — by session",
             xlabel="t-SNE 1", ylabel="t-SNE 2")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_temporal_embedding(
    wf: WindowedFeatures,
    *,
    title_prefix: str = "",
    perplexity: float = 30,
    figsize: tuple[float, float] = (16, 6),
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """t-SNE embedding coloured by time, one subplot per session.

    Useful for seeing how the horse's movement evolves over time
    (e.g. walk → trot transition).

    Parameters
    ----------
    wf : WindowedFeatures
        Combined feature windows.
    """
    X = _preprocess_features(wf.features)
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, len(X) // 4)),
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X)

    sessions = wf.session_ids if wf.session_ids is not None else np.full(len(X), "all")
    unique_sessions = np.unique(sessions)
    n_sess = len(unique_sessions)

    fig, axes = plt.subplots(1, n_sess, figsize=figsize, squeeze=False)

    for i, sid in enumerate(unique_sessions):
        ax = axes[0, i]
        mask = sessions == sid
        pts = X_tsne[mask]
        ts = wf.timestamps_ms[mask]
        # Normalise time to [0,1] for colormap
        t_norm = (ts - ts.min()) / max(ts.max() - ts.min(), 1)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=t_norm, cmap="viridis",
                        s=8, alpha=0.7, edgecolors="none")
        ax.set_title(f"{title_prefix}{sid}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        plt.colorbar(sc, ax=ax, label="time (normalised)")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_color_map(
    unique_vals: np.ndarray, cmap_name: str = "tab10"
) -> dict[str, tuple[float, ...]]:
    cmap = plt.get_cmap(cmap_name)
    return {val: cmap(i % cmap.N) for i, val in enumerate(unique_vals)}


def _scatter(
    ax: plt.Axes,
    points: np.ndarray,
    group: np.ndarray,
    color_map: dict,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    for g in np.unique(group):
        mask = group == g
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            c=[color_map[g]],
            label=str(g),
            s=8,
            alpha=0.6,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(markerscale=3, fontsize=8)
