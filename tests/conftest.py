import sys
from pathlib import Path

import numpy as np
import pytest

from sklearn.datasets import make_blobs


# Ensure repository root is importable (for `students` package in CI/local)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def simple_data():
    # small synthetic 2D blob dataset for testing
    X, y = make_blobs(n_samples=50, centers=3, random_state=0)
    return X, y
