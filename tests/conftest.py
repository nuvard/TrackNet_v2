import numpy as np
import pytest

@pytest.fixture(scope='session', autouse=True)
def set_random_seed():
    # for reproducible results
    np.random.seed(3)