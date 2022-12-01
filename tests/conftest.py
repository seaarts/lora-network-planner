import numpy as np
import pytest


@pytest.fixture
def getGenerators():
    """Get identical rngs for testing randomized functions."""
    seed = 11261432
    rng1 = np.random.default_rng(seed=seed)
    rng2 = np.random.default_rng(seed=seed)
    return rng1, rng2
