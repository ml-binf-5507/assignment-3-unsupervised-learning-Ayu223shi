"""
clustering.py

Wrappers for k-means and hierarchical clustering.  All functions should return
labels and any additional information required for plotting or evaluation.
"""

from typing import Any, Tuple

import numpy as np


def run_kmeans(X: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, Any]:
    """Run k-means clustering and return labels and cluster centers.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    labels : np.ndarray
        Integer cluster labels for each sample.
    centers : Any
        Object representing cluster centers (e.g. np.ndarray).
    """
    raise NotImplementedError("Implement k-means here")


def run_hierarchical(X: np.ndarray, n_clusters: int = 3, linkage: str = "ward") -> np.ndarray:
    """Perform agglomerative clustering and return labels.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_features).
    n_clusters : int
        The number of clusters to find.
    linkage : str
        Linkage criterion for clustering ("ward", "complete", etc.).

    Returns
    -------
    labels : np.ndarray
        Integer labels for each sample.
    """
    raise NotImplementedError("Implement hierarchical clustering here")
