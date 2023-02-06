"""
An randomized epsilon-net algoirthm for computng a Hitting Set.

Given a matrix ``A``, a hitting set for ``A`` is a set cover for ``A.T``.

Based on the algorithm by Nabil Mustafa.

Relies on an existing LP-solution.

Only applicable to set cover instances, in which contributions in ``A`` are
binary, and all deamnds have unit requirements, and facilities unit cost.
"""


def _sampleFromSet(colId, A, weights, gamma, seed=None):
    """
    Sample each point from a given column / set.

    Params
    ------
    colId: int
        The index of column from which to sample.
    A : array_like
        Array of binary contributions.
    weights : array_like
        Vector of weights for each element.
    gamma : float
        A parameter for the aggression of sampling.
    seed : int, optional
        ``Int`` or ``np.random.Generator`` to pass to RNG. Default ``None``.

    Returns
    -------
    sample : set
        Set of sampled element indices.
    """
    pass
