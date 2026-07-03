"""
clustering.py

Wrappers for k-means and hierarchical clustering.  All functions should return
labels and any additional information required for plotting or evaluation.
"""

from sklearn.cluster import KMeans, AgglomerativeClustering

import numpy as np


def run_kmeans(X, n_clusters = 3, random_state = 42) :
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
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, model.cluster_centers_


def run_hierarchical(X, n_clusters = 3, linkage = "ward"):
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
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return labels
