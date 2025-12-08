import sys
import os

import pytest
sys.path.append(os.getcwd())
import numpy as np
from sklearn.datasets import load_iris

@pytest.fixture(scope="session")
def random_seed():
    np.random.seed(0)


@pytest.fixture
def simple_dataset(random_seed):
    """Generate a simple synthetic dataset for binary classification."""
    X = np.random.randn(100, 6)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y

@pytest.fixture
def iris_dataset():
    """Generate a simple synthetic dataset for binary classification."""
    data = load_iris()
    X = data.data
    y = data.target
    return X, y