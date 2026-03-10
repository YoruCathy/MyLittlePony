"""Walk gait detection and prediction.

Provides a detector trained on known walk data that can identify walk
vs. non-walk (e.g. trot) windows in unseen riding sessions.

Two complementary approaches:

1.  **Novelty detection** — trained *only* on walk windows, flags
    anything that deviates as "not walk".  Uses One-Class SVM or
    Isolation Forest.
2.  **Supervised classifier** — once pseudo-labels are available (from
    the novelty detector or manual annotation), trains a Random Forest
    for sharper boundaries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from mlp.analysis.features import WindowedFeatures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-classes for results
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Per-window prediction output.

    Attributes
    ----------
    timestamps_ms : np.ndarray
        Centre timestamps for each window.
    predictions : np.ndarray
        Predicted label per window (``"walk"`` / ``"trot"``).
    scores : np.ndarray
        Decision-function or probability score per window.  Higher →
        more "walk-like".
    """

    timestamps_ms: np.ndarray
    predictions: np.ndarray
    scores: np.ndarray

    @property
    def walk_mask(self) -> np.ndarray:
        return self.predictions == "walk"

    @property
    def trot_mask(self) -> np.ndarray:
        return self.predictions != "walk"

    def summary(self) -> dict[str, Any]:
        n = len(self.predictions)
        n_walk = int(self.walk_mask.sum())
        n_trot = n - n_walk
        return {
            "total_windows": n,
            "walk_windows": n_walk,
            "trot_windows": n_trot,
            "walk_pct": round(100 * n_walk / max(n, 1), 1),
        }


# ---------------------------------------------------------------------------
# Preprocessing helper
# ---------------------------------------------------------------------------

def _build_preprocessor() -> Pipeline:
    """Impute NaN/inf, then standardise."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )


def _clean(X: np.ndarray) -> np.ndarray:
    """Replace non-finite values with NaN for imputer."""
    X = np.array(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    return X


# ---------------------------------------------------------------------------
# Novelty-based walk detector
# ---------------------------------------------------------------------------

class WalkNoveltyDetector:
    """Detects walk gait via novelty detection.

    Trained exclusively on *walk* windows — at inference any window
    that looks sufficiently different is labelled as ``"trot"``
    (or more generally "not walk").

    Parameters
    ----------
    method : ``"ocsvm"`` | ``"iforest"``
        Which algorithm to use.
    contamination : float
        Expected fraction of anomalies (non-walk) in the *training*
        set.  Set to a small value (e.g. 0.01) when training data is
        clean walk.
    **kwargs
        Forwarded to the underlying scikit-learn estimator.
    """

    def __init__(
        self,
        method: Literal["ocsvm", "iforest"] = "iforest",
        contamination: float = 0.01,
        **kwargs: Any,
    ) -> None:
        self.method = method
        if method == "ocsvm":
            self._model = OneClassSVM(
                kernel="rbf",
                nu=contamination,
                gamma="scale",
                **kwargs,
            )
        elif method == "iforest":
            self._model = IsolationForest(
                contamination=contamination,
                n_estimators=200,
                random_state=42,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")
        self._preprocessor = _build_preprocessor()
        self._is_fitted = False

    def fit(self, walk_features: WindowedFeatures) -> "WalkNoveltyDetector":
        """Fit on known walk windows.

        Parameters
        ----------
        walk_features : WindowedFeatures
            Feature windows that are known to be walk.
        """
        X = _clean(walk_features.features)
        X = self._preprocessor.fit_transform(X)
        np.clip(X, -10, 10, out=X)
        self._model.fit(X)
        self._is_fitted = True
        n = len(X)
        logger.info("WalkNoveltyDetector(%s) fitted on %d walk windows", self.method, n)
        return self

    def predict(self, wf: WindowedFeatures) -> PredictionResult:
        """Predict walk / not-walk for each window.

        Returns
        -------
        PredictionResult
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted — call .fit() first")
        X = _clean(wf.features)
        X = self._preprocessor.transform(X)
        np.clip(X, -10, 10, out=X)

        # decision_function: positive = inlier (walk), negative = outlier
        scores = self._model.decision_function(X)
        raw_preds = self._model.predict(X)  # +1 = inlier, -1 = outlier

        labels = np.where(raw_preds == 1, "walk", "trot")
        return PredictionResult(
            timestamps_ms=wf.timestamps_ms.copy(),
            predictions=labels,
            scores=scores,
        )


# ---------------------------------------------------------------------------
# Supervised walk classifier
# ---------------------------------------------------------------------------

class WalkClassifier:
    """Supervised walk-vs-trot classifier (Random Forest).

    Can be trained with explicit labels or with pseudo-labels from the
    novelty detector.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    """

    def __init__(self, n_estimators: int = 300, random_state: int = 42) -> None:
        self._clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self._preprocessor = _build_preprocessor()
        self._is_fitted = False

    def fit(
        self,
        features: WindowedFeatures,
        labels: np.ndarray | None = None,
    ) -> "WalkClassifier":
        """Train the classifier.

        Parameters
        ----------
        features : WindowedFeatures
            Training windows.
        labels : np.ndarray | None
            Per-window labels (``"walk"`` / ``"trot"``).  If *None*,
            uses ``features.labels``.
        """
        y = labels if labels is not None else features.labels
        if y is None:
            raise ValueError("No labels provided")
        X = _clean(features.features)
        X = self._preprocessor.fit_transform(X)
        np.clip(X, -10, 10, out=X)
        self._clf.fit(X, y)
        self._is_fitted = True

        # Quick cross-val estimate
        cv_scores = cross_val_score(self._clf, X, y, cv=5, scoring="f1_macro")
        logger.info(
            "WalkClassifier fitted — 5-fold F1: %.3f ± %.3f",
            cv_scores.mean(), cv_scores.std(),
        )
        return self

    def predict(self, wf: WindowedFeatures) -> PredictionResult:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted — call .fit() first")
        X = _clean(wf.features)
        X = self._preprocessor.transform(X)
        np.clip(X, -10, 10, out=X)

        preds = self._clf.predict(X)
        proba = self._clf.predict_proba(X)
        # Score = probability of "walk" class
        walk_idx = list(self._clf.classes_).index("walk")
        scores = proba[:, walk_idx]

        return PredictionResult(
            timestamps_ms=wf.timestamps_ms.copy(),
            predictions=preds,
            scores=scores,
        )

    @property
    def feature_importances(self) -> np.ndarray:
        return self._clf.feature_importances_

    def top_features(self, feature_names: list[str], n: int = 15) -> list[tuple[str, float]]:
        """Return the *n* most important features by Gini importance."""
        imp = self.feature_importances
        idx = np.argsort(imp)[::-1][:n]
        return [(feature_names[i], float(imp[i])) for i in idx]
