"""
dimensionality_reduction.py

Students implement three wrappers that apply PCA, t-SNE, and UMAP to a feature
matrix X and return the transformed data.
"""

from typing import Any

import numpy as np


def apply_pca(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Perform PCA on the input array.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    n_components : int, default=2
        Number of principal components to keep.
    random_state : int, default=42
        Random seed for reproducibility (if needed by the implementation).

    Returns
    -------
    np.ndarray
        Transformed data with shape (n_samples, n_components).
    """
    raise NotImplementedError("Implement PCA transformation here")


def apply_tsne(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Apply t-SNE to the data.

    The sklearn.manifold.TSNE class may be used.  The function should return the
    2‑D embedding.
    """
    raise NotImplementedError("Implement t-SNE transformation here")


def apply_umap(X: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Apply UMAP to the data.

    Requires the `umap-learn` package.  Return an array of shape
    (n_samples, n_components).
    """
    raise NotImplementedError("Implement UMAP transformation here")
