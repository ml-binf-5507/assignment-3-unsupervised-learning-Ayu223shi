"""
dimensionality_reduction.py

Students implement three wrappers that apply PCA, t-SNE, and UMAP to a feature
matrix X and return the transformed data.
"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np


def apply_pca(X, n_components = 2, random_state = 42):
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
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X)

def apply_tsne(X, n_components = 2, random_state = 42):
    """Apply t-SNE to the data.

    The sklearn.manifold.TSNE class may be used.  The function should return the
    2‑D embedding.
    """
    tsne = TSNE(n_components=n_components, random_state=random_state)
    return tsne.fit_transform(X)

def apply_umap(X, n_components = 2, random_state = 42):
    """Apply UMAP to the data.

    Requires the `umap-learn` package.  Return an array of shape
    (n_samples, n_components).
    """
    umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
    return umap_model.fit_transform(X)
