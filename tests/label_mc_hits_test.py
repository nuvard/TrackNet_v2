import numpy as np
import pytest

from scipy.spatial import KDTree

from src.label_mc_hits import euclidean
from src.label_mc_hits import brute_force_1nn

def test_brute_force_1nn():
    keys = np.random.normal(size=(10000, 3))
    queries = np.random.normal(size=(1000, 3))
    nearest = brute_force_1nn(keys, queries)
    # create KDtree
    scipy_nearest = KDTree(keys).query(queries)[1]
    # assert
    np.testing.assert_equal(nearest, scipy_nearest)

def test_euclidean():
    x = np.random.normal(size=(10000, 100))
    y = np.random.normal(size=(10000, 100))
    dist = euclidean(x, y)
    # square, because for computations reasons
    # euclidean() doesn't use square root
    numpy_dist = np.linalg.norm(x-y, axis=1)**2
    # assert
    np.testing.assert_allclose(dist, numpy_dist)

if __name__ == '__main__':
    pytest.main([__file__])