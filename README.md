[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YVSugOT1)
# Assignment 3 – Dimensionality Reduction and Clustering

## 🎯 Goal

To independently explore how feature representation influences clustering results using:

- PCA
- t-SNE
- UMAP
- K-Means clustering
- Hierarchical clustering

You will generate visualizations that allow comparison of how different representations affect the apparent structure of the data.

## 🧩 Dataset Description

This assignment uses the **Breast Cancer Wisconsin (Diagnostic)** dataset, available through **scikit-learn**.

The dataset contains measurements computed from digitized images of fine‑needle aspirates of breast masses. Each observation corresponds to a tumor sample, and the features describe characteristics of the cell nuclei present in the image.

**Key Dataset Properties**

- Number of samples: 569
- Number of features: 30
- Feature types: Continuous numeric measurements
- Target labels:
  - Benign (non-cancerous)
  - Malignant (cancerous)

The features capture properties such as radius, texture, perimeter, area, smoothness, and concavity.

> **Note:** The diagnostic label is **not used to train any model**. It is reserved for visualization and interpretation, allowing you to evaluate how well unsupervised methods recover clinically meaningful structure.

## 🔄 Analytical Workflow

Your analysis will proceed in three stages:

1. **Feature Representation and Visualization**
   - Apply dimensionality reduction techniques to project the high-dimensional feature space into two dimensions:
     - PCA
     - t-SNE
     - UMAP

2. **Clustering in the Full Feature Space**
   - Perform clustering using the original 30-feature dataset:
     - K-Means clustering
     - Hierarchical clustering
   - Select one clustering method for visualization in the next step.

3. **Visualization Requirements**
   - Produce two figures, each consisting of three subplots (1×3 layout).

   **Figure 1 — Representation Comparison (True Labels)**
   
   Create a single figure with three panels:
   - PCA projection
   - t-SNE projection
   - UMAP projection

   In each subplot:
   - Each point represents a sample
   - Points are coloured by true diagnostic label (benign vs malignant)

   **Figure 2 — Cluster Comparison**

   Using the same three projections (PCA, t-SNE, UMAP), create a second figure with three panels where:
   - Each point represents a sample
   - Points are coloured by cluster assignment from your chosen clustering method

   This comparison will allow you to evaluate:
   - Whether clusters align with clinical outcomes
   - Whether the representation exaggerates or obscures structure
   - The extent to which unsupervised learning recovers meaningful groupings

## ✅ Autograding & Verification

Automated tests run when you push your code. The autograder verifies:

- Dimensionality reduction methods executed
- Clustering algorithms implemented
- Required plots generated

The GitHub component is not graded for code style, optimization, or efficiency. It is used to verify that:

- You independently implemented the required models
- The plots uploaded to Blackboard were generated from your own code

If the GitHub Classroom component is not completed:

- Marks associated with plot uploads and related short-answer questions may be deducted or withheld.
- Blackboard submissions containing plots without corresponding verified execution in GitHub may not receive credit.

---

## 🧱 Repository structure

```text
assn3-unsupervised-learning/
├── .github/
│   ├── workflows/classroom.yml        # CI workflow for autograding
│   └── classroom/autograding.json    # test configuration
├── students/                         # code templates for students
│   ├── dimensionality_reduction.py
│   ├── clustering.py
│   └── visualization.py
├── tests/                            # pytest tests used by CI
│   ├── conftest.py
│   └── test_submission.py
├── notebooks/                        # analysis notebook (optional)
├── data/                             # dataset or instructions
├── requirements.txt                  # Python dependencies
├── validate_submission.py            # helper
└── README.md                         # this file
```