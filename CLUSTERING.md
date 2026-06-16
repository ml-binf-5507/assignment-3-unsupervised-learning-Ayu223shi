# Clustering Quick Reference

Essential clustering algorithms and evaluation metrics for unsupervised learning.

## K-Means Clustering

### Basic Usage

```python
from sklearn.cluster import KMeans

# Create and fit k-means
kmeans = KMeans(
    n_clusters=3,        # Number of clusters
    random_state=42,     # Seed for reproducibility
    n_init=10,           # Number of initializations (default: 10)
    max_iter=300         # Max iterations per initialization
)

# Fit to data
labels = kmeans.fit_predict(X)

# Get cluster centers
centers = kmeans.cluster_centers_  # shape: (n_clusters, n_features)

# Inertia (within-cluster sum of squares)
inertia = kmeans.inertia_
```

### Finding Optimal k (Elbow Method)

```python
inertias = []
silhouette_scores = []
k_range = range(2, 11)

from sklearn.metrics import silhouette_score

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# Plot both
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(k_range, silhouette_scores, 'go-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')

plt.tight_layout()
plt.show()
```

### Key Characteristics

- **Fast** – O(n) per iteration, very scalable
- **Centroid-based** – finds cluster centers (means)
- **Spherical clusters** – assumes roughly equal-sized, spherical clusters
- **Local optima** – may not find global best solution (use multiple initializations)
- **Requires k** – must specify number of clusters in advance

---

## Agglomerative Clustering (Hierarchical)

### Basic Usage

```python
from sklearn.cluster import AgglomerativeClustering

# Create hierarchical clustering
hierarchical = AgglomerativeClustering(
    n_clusters=3,           # Number of clusters to cut at
    linkage='ward'          # Linkage criterion (see below)
)

# Fit and get labels
labels = hierarchical.fit_predict(X)
```

### Linkage Criteria

```python
# Different linkage methods measure distance between clusters differently

# 'ward': minimizes variance (similar to k-means objective)
agg_ward = AgglomerativeClustering(n_clusters=3, linkage='ward')

# 'complete': maximum distance between clusters (conservative)
agg_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')

# 'average': mean distance between clusters
agg_average = AgglomerativeClustering(n_clusters=3, linkage='average')

# 'single': minimum distance between clusters (can create chains)
agg_single = AgglomerativeClustering(n_clusters=3, linkage='single')
```

### Dendrogram Visualization (Optional)

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Create linkage matrix
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.axhline(y=threshold, color='r', linestyle='--', label='Cut threshold')
plt.legend()
plt.show()
```

### Key Characteristics

- **Hierarchical** – creates a tree of clusters at different levels
- **Deterministic** – same result every time (no randomness)
- **Flexible** – can choose different linkage criteria
- **Interpretable** – dendrogram shows cluster merging process
- **Slower** – O(n²) or O(n³) depending on implementation

---

## Clustering Evaluation Metrics

### Silhouette Score

Measures how similar a point is to its own cluster vs. other clusters.

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Overall silhouette score (-1 to 1, higher is better)
score = silhouette_score(X, labels)

# Per-sample silhouette scores
scores_per_sample = silhouette_samples(X, labels)

# Visualize silhouette values by cluster
fig, ax = plt.subplots(figsize=(8, 6))
y_lower = 10

for i in range(n_clusters):
    cluster_silhouette_vals = scores_per_sample[labels == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                      0, cluster_silhouette_vals,
                      alpha=0.7)
    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster Label')
ax.set_title('Silhouette Plot for Each Cluster')
ax.axvline(x=score, color='red', linestyle='--', label=f'Average: {score:.3f}')
plt.show()
```

### Davies-Bouldin Index (For Reference)

Measures average similarity between clusters (lower is better).

```python
from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X, labels)
# Lower values indicate better separation (unlike silhouette)
```

### Inertia

Sum of squared distances from each point to its cluster center.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
inertia = kmeans.inertia_
```

### Calinski-Harabasz Index (For Reference)

Ratio of between-cluster to within-cluster variance (higher is better).

```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X, labels)
```

---

## Full Example: Grid Search Over Hyperparameters

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

# Grid of k values to test
k_values = [2, 3, 4, 5, 6, 7]
results = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    silhouette = silhouette_score(X, labels)
    inertia = kmeans.inertia_
    
    results.append({
        'k': k,
        'silhouette': silhouette,
        'inertia': inertia
    })

results_df = pd.DataFrame(results)
print(results_df)

# Find best k by silhouette score
best_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']
print(f"Best k: {best_k}")
```

---

## Comparison

| Aspect | K-Means | Hierarchical |
|--------|---------|--------------|
| **Type** | Centroid-based | Hierarchical |
| **Speed** | Fast | Slower |
| **Deterministic** | No (random init) | Yes |
| **Requires k** | Yes | Can be flexible |
| **Cluster shape** | Spherical | Any |
| **Interpretability** | Medium | High (dendrogram) |

---

## Common Pitfalls

- **No ground truth** – without labels, cluster quality is subjective
- **Scaling matters** – both k-means and hierarchical use distance; scale features first
- **Choosing k** – use elbow method or silhouette analysis, but also rely on domain knowledge
- **Outliers** – both algorithms are sensitive to outliers; consider preprocessing
- **High dimensions** – clustering in high dimensions can be unreliable; reduce dimensionality first

---

## Resources

- [K-Means Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Hierarchical Clustering Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [Clustering Metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
