import json
import os

import numpy as np
import pytest

from students import dimensionality_reduction, clustering, visualization


EXPECTED_PLOTS = [
    "figure1.png",  # representation comparison (true labels)
    "figure2.png",  # cluster comparison (chosen method)
]


def test_function_signatures():
    # ensure required callables exist
    assert hasattr(dimensionality_reduction, "apply_pca")
    assert hasattr(dimensionality_reduction, "apply_tsne")
    assert hasattr(dimensionality_reduction, "apply_umap")
    assert hasattr(clustering, "run_kmeans")
    assert hasattr(clustering, "run_hierarchical")
    assert hasattr(visualization, "plot_2d")
    # new helpers for multi-panel figures
    assert hasattr(visualization, "plot_representation_comparison")
    assert hasattr(visualization, "plot_cluster_comparison")


def test_pipeline_completes(simple_data, tmp_path):
    X, y = simple_data

    # apply reductions
    pca = dimensionality_reduction.apply_pca(X)
    tsne = dimensionality_reduction.apply_tsne(X)
    umap = dimensionality_reduction.apply_umap(X)

    # shapes
    assert pca.shape[0] == X.shape[0] and pca.shape[1] == 2
    assert tsne.shape == pca.shape
    assert umap.shape == pca.shape

    # clustering
    labels_k, centers = clustering.run_kmeans(pca, n_clusters=3)
    assert labels_k.shape[0] == X.shape[0]
    labels_h = clustering.run_hierarchical(pca, n_clusters=3)
    assert labels_h.shape[0] == X.shape[0]

    # multi-panel figures per instructions (single-view plots are not required)
    visualization.plot_representation_comparison(
        pca, tsne, umap, y, filename=str(tmp_path / "figure1.png")
    )
    visualization.plot_cluster_comparison(
        pca, tsne, umap, labels_k, filename=str(tmp_path / "figure2.png")
    )

    for fname in EXPECTED_PLOTS:
        assert (tmp_path / fname).exists(), f"{fname} was not created"
        assert (tmp_path / fname).stat().st_size > 0

    # metrics JSON remains optional
    results = {
        "kmeans": {"n_clusters": int(np.unique(labels_k).size)},
        "hierarchical": {"n_clusters": int(np.unique(labels_h).size)},
    }
    json_path = tmp_path / "clustering_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f)

    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert "kmeans" in data and "hierarchical" in data

