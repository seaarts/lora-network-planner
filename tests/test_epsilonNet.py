import numpy as np
import pytest

from loraplan.algorithms import _sampleEpsilonNet, _unhitSet, sampleEpsilonNet


@pytest.mark.parametrize(
    "w, e, d, p, delta",
    [
        ([-1, 0], -0.2, 5, [0, 1], 0.5),  # negative weights
        ([1, 0], -0.2, 5, [0, 1], 0.5),  # negative e
        ([1, 1], 0, 5, [0, 1], 0.5),  # zero e
        ([1, 1], 0, -1, [0, 1], 0.5),  # negative vc-dim
        ([1, 1], 0, 0, [0, 1], 0.5),  # zero vc-dim
        ([1, 1], 0, 5, [0, 1], 0),  # zero delta
        ([1, 1], 0, 5, [0, 1], -0.5),  # negative delta
        ([1, 1, 1], 0, 5, [0, 1], 0.5),  # weight-points mismatch
    ],
)
def test_epsilonNetBadInput(w, e, d, p, delta):
    """Test e-net finder for bad inputs.

    This is the non-underscored version that checks for valid inputs,
    and makes arrays into ndarrays. It should raise value errors for
    various badly formed inputs, but should accept array_like inputs,
    and not just ndarrays.
    """
    with pytest.raises(ValueError):
        sampleEpsilonNet(w, e, d, p, delta)


@pytest.mark.parametrize(
    "w, e, d, p, delta",
    [
        ([1, 1], 0.1, 5, [0, 1], 0.5),
        ([1, 2, 4], 0.1, 5, [0, 1, 2], 0.5),
        ([1, 2, 3, 4, 5, 6], 0.1, 5, [3, 4, 5, 6, 7, 8], 0.5),
    ],
)
def test_sampleEpsilonNet(w, e, d, p, delta):
    """Test e-net finder on good inputs.

    This is a minimal test ensuring a subset of the given points is returned.
    The inputs here are assumed to have passed `sampleEpsilonNet` checks.
    """
    w, p = np.array(w), np.array(p)
    net = set(_sampleEpsilonNet(w, e, d, p, delta=delta))
    pts = set(p)
    assert net.issubset(pts)


@pytest.mark.parametrize(
    "w, e, d, p, delta, expected",
    [
        ([1, 0], 0.01, 1000, [0, 1], 0.01, [0]),
        ([1, 0, 0], 0.01, 1000, [0, 1, 2], 0.01, [0]),
        ([1, 0, 0, 0, 0, 0], 0.01, 1000, [3, 4, 5, 6, 7, 8], 0.01, [3]),
    ],
)
def test_sampleEpsilonNetSingletons(w, e, d, p, delta, expected):
    """A minimal test ensuring correct singleton is returned.

    Tests whether a very heavily weighted singleton is returned. While, the
    algorithm is randomized, if the "score" of an item exceeds 1 it is selected
    with certainty. Meanwhile a weight of 0 means an item is never selected.
    This way, we can force selection of given singletons. We test for this.
    """
    w, p = np.array(w), np.array(p)
    assert _sampleEpsilonNet(w, e, d, p, delta=delta) == expected


"""Test the e-net finder for good non-ndarray input."""


@pytest.mark.parametrize(
    "w, e, d, p, delta",
    [
        ([1, 1], 0.1, 5, [0, 1], 0.5),
        ([1, 2, 4], 0.1, 5, [0, 1, 2], 0.5),
        ([1, 2, 3, 4, 5, 6], 0.1, 5, [3, 4, 5, 6, 7, 8], 0.5),
    ],
)
def test_sampleEpsilonNetNonNumpy(w, e, d, p, delta):
    """A minimal test ensuring a subset is returned.

    Repeats earler test but for the underscored version.
    Note that we pass array_like (lists) and not ndarrays.
    """

    net = set(sampleEpsilonNet(w, e, d, p, delta=delta))
    pts = set(p)
    assert net.issubset(pts)


@pytest.mark.parametrize(
    "w, e, d, p, delta, expected",
    [
        ([1, 0], 0.01, 1000, [0, 1], 0.01, [0]),
        ([1, 0, 0], 0.01, 1000, [0, 1, 2], 0.01, [0]),
        ([1, 0, 0, 0, 0, 0], 0.01, 1000, [3, 4, 5, 6, 7, 8], 0.01, [3]),
    ],
)
def test_sampleEpsilonNetSingletonsNonNumpy(w, e, d, p, delta, expected):
    """A minimal test ensuring correct singleton is returned.

    Repat analogous test from `_sampleEpsilonNet`.
    """
    assert sampleEpsilonNet(w, e, d, p, delta=delta) == expected


"""Test unhit set finder."""


@pytest.mark.parametrize(
    "A, x, expected",
    [
        (np.array([[1, 1], [1, 1]]), np.array([0, 1]), []),
        (np.array([[1, 1], [1, 1]]), np.array([0, 0]), [0, 1]),
        (np.array([[1, 1], [1, 0]]), np.array([0, 1]), [1]),
        (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([1, 0, 0]), [1, 2]),
    ],
)
def test_unhitSet(A, x, expected):
    assert _unhitSet(A, x) == expected
