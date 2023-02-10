"""
An randomized epsilon-net algoirthm for computng a Hitting Set.

Given a an element-set indicednce matrix ``A``, a hitting set for ``A`` is a
set cover for ``A.T``. Each row in ``A`` corresponds to a set; each column to
an element. Hence the matrix-vector product ``Ax`` gives the number of hits
for each set. The selection ``x`` encodes a hitting set if ``Ax >= 1``.

This algorithm is based on the algorithm by Nabil Mustafa.

Relies on an existing LP-solution.

Only applicable to set cover instances, in which contributions in ``A`` are
binary, and all deamnds have unit requirements, and facilities unit cost.
"""

import numpy as np


def _sampleEpsilonNet(weights, epsilon, d, points, delta=0.5, seed=None):
    """
    Sample points independently probability propotional to weights.

    Parameters
    ----------
    weights : ndarray
        ``(nPoints, )`` array of non-negative weights.
    epsilon : float
        Constant in (0, 1) that sets the weight of sets to be hit by the net.
    d : float
        The VC-dimension.
    points: ndarray
        ``(nPoints, )`` array of pint indices.
    delta : float
        A scaling parameter for the sampling probability.
    seed : int, optional
        ``int`` or ``np.random.Generator`` to pass to RNG. Default ``None``.

    Returns
    -------
    sample : list
        list of sampled element indices.
    """

    rng = np.random.default_rng(seed=seed)

    # normalize weights, evaluate two parts of probability
    measure = weights / np.sum(weights)
    constant = 4 * np.log(2 / delta)
    probs = 8 * d * np.log(13 / epsilon)

    # form valid probabilities
    probability = weights * np.maximum(constant, probs) / epsilon
    probability = np.minimum(1, probability)

    # sample and return
    U = rng.uniform(size=len(measure))
    return list(points[U < measure])


def sampleEpsilonNet(weights, epsilon, d, points, delta=0.5, seed=None):
    """
    Sample a likely epsilon-net.

    This is used as an epsilon-net finder algoirthm. The parameter
    ``delta`` controls the aggressivness of the sampling; higher values
    increase the probability the output is an epsilon-net, at
    the cost of a higher expected number of points.

    This implementation is based on Blumer et al. Theorem A2.2.
    There may be potential to improve on the constants.

    Parameters
    ----------
    weights : ndarray
        ``(nPoints, )`` array of non-negative weights.
    epsilon : float
        Constant in (0, 1) that sets the weight of sets to be hit by the net.
    d : float
        The VC-dimension.
    points: ndarray
        ``(nPoints, )`` array of pint indices.
    delta : float
        A scaling parameter for the sampling probability.
    seed : int, optional
        ``int`` or ``np.random.Generator`` to pass to RNG. Default ``None``.

    Returns
    -------
    sample : list
        list of sampled element indices.
    """
    weights = np.array(weights)
    points = np.array(points)

    if not np.all(weights >= 0):
        raise ValueError("weights must be non-negative.")

    if epsilon <= 0:
        raise ValueError("epsilon must be a positive scalar.")

    if d <= 0:
        raise ValueError("VC-dimension d must be positive.")

    if delta <= 0:
        raise ValueError("delta must be positive.")

    if weights.shape != points.shape:
        raise ValueError("weights and points must be of same shape.")

    return _sampleEpsilonNet(weights, epsilon, d, points, delta, seed=seed)


def _unhitSet(A, x):
    """
    Returns a list of indices of unhit sets.

    Different implementations algorithms can use different
    rules to choose which points to include. If the list
    is empty, algorithms should terminate.

    Parameters
    ----------
    A : ndarray
        Element-set incidence matrix of size (nPoints, nSets).
    x : ndarray
        Binary vector indicating element selections.

    Returns
    -------
    unhits : list
        List of indices of points unhit by x.
    """
    hits = A @ x

    return list(np.where(hits == 0)[0])
