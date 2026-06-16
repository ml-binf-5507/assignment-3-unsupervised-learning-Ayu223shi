# Dimensionality Reduction Quick Reference

Essential techniques for reducing high-dimensional data to 2D/3D for visualization and clustering.

## PCA (Principal Component Analysis)

### Basic Usage

```python
from sklearn.decomposition import PCA
import numpy as np

# Create PCA transformer
pca = PCA(n_components=2, random_state=42)

# Fit and transform data
X_pca = pca.fit_transform(X)

# X_pca now has shape (n_samples, 2)
```

### Interpreting Results

```python
# Variance explained by each component
print(pca.explained_variance_ratio_)  # e.g. [0.60, 0.25]

# Cumulative variance
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"First 2 components explain {cumsum_var[1]:.1%} of variance")

# PCA loadings (how much each original feature contributes)
loadings = pca.components_  # shape: (n_components, n_features)
```

### Key Characteristics

- **Linear** transformation that preserves variance
- **Fast** – good for exploratory analysis
- **Interpretable** – components are linear combinations of original features
- **Best when:** clusters are linearly separable or data follows Gaussian distribution

---

## t-SNE (t-Distributed Stochastic Neighbor Embedding)

### Basic Usage

```python
from sklearn.manifold import TSNE

# Create t-SNE transformer
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,  # Tune based on dataset size (typically 5-50)
    n_iter=1000     # Number of optimization iterations
)

# Fit and transform
X_tsne = tsne.fit_transform(X)
```

### Interpreting Results

```python
# t-SNE has no parameters like PCA to extract afterward
# The output coordinates are the only result
# Distance between points is meaningful locally but NOT globally
```

### Key Characteristics

- **Nonlinear** – excellent at revealing local cluster structure
- **Slow** – may take seconds to minutes on large datasets
- **Stochastic** – results vary between runs; use `random_state` to fix
- **Local structure preserved** – distances between nearby points are meaningful
- **Global structure NOT preserved** – distance between distant clusters is not reliable
- **Best when:** you want to visualize clusters (not for downstream ML tasks)

---

## UMAP (Uniform Manifold Approximation and Projection)

### Basic Usage

```python
import umap

# Create UMAP transformer
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=15,    # Size of local neighborhoods (default: 15)
    min_dist=0.1       # Minimum distance between points (default: 0.1)
)

# Fit and transform
X_umap = reducer.fit_transform(X)
```

### Tuning Parameters

```python
# Larger n_neighbors -> smoother, more global structure
# Smaller n_neighbors -> more local detail

# Larger min_dist -> points spread out more
# Smaller min_dist -> points can be closer (finer structure)

# Example: For large dataset, increase n_neighbors for speed
reducer = umap.UMAP(n_components=2, n_neighbors=30, random_state=42)
```

### Key Characteristics

- **Nonlinear** – similar to t-SNE but preserves more global structure
- **Faster** than t-SNE – practical for larger datasets
- **Parametric** – can be applied to new data (unlike t-SNE)
- **Flexible** – tunable parameters let you control local vs. global emphasis
- **Best when:** you want both local clusters AND global structure, or need speed

---

## Comparison

| Aspect | PCA | t-SNE | UMAP |
|--------|-----|-------|------|
| **Type** | Linear | Nonlinear | Nonlinear |
| **Speed** | Very fast | Slow | Medium |
| **Local structure** | No | Excellent | Excellent |
| **Global structure** | Preserved | Not reliable | Mostly preserved |
| **Reproducible** | Yes | No (set seed) | Yes (set seed) |
| **Use case** | Exploratory / preprocessing | Visualization only | Visualization + downstream ML |

---

## Full Example: Reduce and Visualize

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# Load data
X = load_your_data()

# Apply three reductions
pca_data = PCA(n_components=2, random_state=42).fit_transform(X)
tsne_data = TSNE(n_components=2, random_state=42).fit_transform(X)
umap_data = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

# Plot side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, data, title in zip(axes, [pca_data, tsne_data, umap_data], 
                            ['PCA', 't-SNE', 'UMAP']):
    ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

plt.tight_layout()
plt.show()
```

---

## Common Pitfalls

- **Scaling**: Always scale your data before PCA (StandardScaler)
- **t-SNE perplexity**: For small datasets (<100 samples), use smaller perplexity (e.g., 5–10)
- **UMAP speed**: If slow, increase `n_neighbors` to 30–50 for faster, coarser embedding
- **Comparing distances**: Only compare distances *within* t-SNE/UMAP plots, not across different methods
- **Randomness**: Set `random_state` for reproducibility

---

## Resources

- [PCA documentation](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca)
- [t-SNE documentation](https://scikit-learn.org/stable/modules/manifold.html#t-sne)
- [UMAP GitHub](https://github.com/lmcinnes/umap)
