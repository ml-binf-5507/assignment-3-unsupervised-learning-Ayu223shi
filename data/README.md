This directory may store a dataset for the assignment.  For the template
we recommend using one of scikit-learn's built-in collections.  For example:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
```

Students are free to substitute any other dataset (e.g. `make_blobs`).  The
autograder only checks that your code executed and produced the expected
output files; it does *not* verify particular cluster structure.
