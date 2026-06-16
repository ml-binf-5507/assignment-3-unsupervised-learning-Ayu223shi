"""
visualization.py

Functions for generating and saving scatter plots of 2-D embeddings and
cluster assignments.  The plotting functions should save PNG files with the
provided filename.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_2d(X: np.ndarray, labels: np.ndarray = None, centers: np.ndarray = None, title: str = "", filename: str = "plot.png") -> None:
    """Create a 2D scatter plot of `X` and save to disk.

    - `X` is assumed to have shape (n_samples, 2).
    - If `labels` is not None, color by label.
    - If `centers` is provided, plot them as larger markers.

    The figure must be closed before returning to avoid resource leaks.
    """
    plt.figure(figsize=(6, 5))
    if labels is None:
        plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7)
    else:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="tab10", alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="cluster", loc="best")
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c="black", s=50, marker="x")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_representation_comparison(
    pca: np.ndarray,
    tsne: np.ndarray,
    umap: np.ndarray,
    true_labels: np.ndarray,
    filename: str = "figure1.png",
) -> None:
    """Create a 1x3 figure showing the three embeddings coloured by true labels.

    Parameters
    ----------
    pca, tsne, umap : np.ndarray
        Arrays of shape (n_samples, 2) containing the two-dimensional
        projections.
    true_labels : np.ndarray
        Ground-truth labels used for colouring (e.g. benign vs malignant).
    filename : str
        Output file path for the saved figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, name in zip(
        axes, (pca, tsne, umap), ("PCA", "t-SNE", "UMAP")
    ):
        sc = ax.scatter(data[:, 0], data[:, 1], c=true_labels, cmap="tab10", s=10, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    fig.colorbar(sc, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def plot_cluster_comparison(
    pca: np.ndarray,
    tsne: np.ndarray,
    umap: np.ndarray,
    cluster_labels: np.ndarray,
    filename: str = "figure2.png",
) -> None:
    """Create a 1x3 figure showing the embeddings coloured by cluster labels.

    Parameters
    ----------
    pca, tsne, umap : np.ndarray
        Arrays of shape (n_samples, 2) containing the two-dimensional
        projections.
    cluster_labels : np.ndarray
        Labels produced by the chosen clustering algorithm.
    filename : str
        Output file path for the saved figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, name in zip(
        axes, (pca, tsne, umap), ("PCA", "t-SNE", "UMAP")
    ):
        sc = ax.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap="tab10", s=10, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    fig.colorbar(sc, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
