# Reference Notes: Dimensionality Reduction & Clustering

## PCA
- linear projection that preserves variance
- computed via SVD
- works well when clusters are linearly separable

## t-SNE
- nonlinear embedding optimized for local neighbourhoods
- perplexity parameter controls effective neighbourhood size
- results may vary run-to-run, use `random_state` to fix

## UMAP
- similar to t-SNE but faster and preserves more global structure
- requires `umap-learn` package
- parameters `n_neighbors` and `min_dist` tune granularity

## K-Means
- centroid-based algorithm
- sensitive to initialization; sklearn uses `k-means++` by default
- objective: minimize within-cluster variance

## Agglomerative Clustering
- bottom-up hierarchical clustering
- linkage options: ``ward``, ``complete``, ``average``
- returns a flat clustering when you cut the dendrogram at a given level

### Evaluation Metrics (not graded)
- silhouette score measures separation between clusters
- inertia (sum of squared distances to centroids) used by k-means

For detailed documentation see scikit-learn's online reference:
https://scikit-learn.org/stable/modules/clustering.html
