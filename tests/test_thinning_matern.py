import numpy as np
import pytest

from loraplan.probability import thinning_matern as thm


@pytest.mark.parametrize(
    "matern_input, expected",
    [
        ((np.zeros((1, 2)), 1), np.repeat(True, 1)),
        ((np.zeros((2, 2)), 1), np.repeat(False, 2)),
        ((np.zeros((3, 2)), 1), np.repeat(False, 3)),
        ((np.array([[0, 0], [1, 1]]), np.array([0.5, 0.5])), np.repeat(True, 2)),
        ((np.array([[0, 0], [1, 1.1], [1, 0.9]]), 0.5), [True, False, False]),
        (([0, 0], 1), [True]),
        (([[0, 0], [0, 1]], 2), [False, False]),
        (([[0, 0, 0], [0, 1, 1]], 2), [False, False]),
        (([[1], [2], [3]], [0.6, 0.6, 0.1]), [False, False, True]),
        (([], 1), []),
    ],
)
def test_maternThinningI(matern_input, expected):
    """Test matern thinning.

    `maternThinningI` takes an array of points and an array or single float
    of radii. It returns a boolean array. Use `np.all()` to test all entries
    are as expected.
    """
    assert np.all(thm.maternThinningI(*matern_input) == expected)
