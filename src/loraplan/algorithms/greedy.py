"""
A greedy algorithm for network planning.

This algorithm iteratviely chooses the facility with the lowest cost per
total coverage. After each selection, the residual requirements and
contributions are computed. The algorihm is relatively fast, even if
we use arrays and not necessarily optimized data strucures for the updates.
"""

import numpy as np


def _greedy_update(r_resid, A_resid, f_fixed, unbuilt):
    """
    Chooses the highest value facility.

    This algorithm implements a greedy step update for Covering Integer
    Programs (CIPs). It takes a vector of resiudal demands ``r_resid``, a
    matrix of residual contributions ``A_resid``, a fixed cost vector
    ``f_fixed``, and a set of indices of unbuilt facilities. It returns the
    index of an index that minimizes the cost per residual coverage, as well
    as updated residual values, contrbutions, and unbilt indices.

    Parameters
    ----------
    r_resid : array_like
        (nDems, ) vector of residual requirements for each demand.
    A_resid : array_like
        (nDems, nFacs) matrix of residual contributions.
    f_fixed : array_like
        (n_facs, ) vector of fixed facility costs.
    unbuild : set
        Set of indices of unbuild facilities.

    Returns
    -------
    reqs_new : np.array
        (nDems, ) vector of updated residual requirements.
    A_new : np.array
        (nDems, nFacs) matrix of updated residual contribtions.
    unbuild_new : set
        Updated set of unbuilt indices.
    facility : int
        Index of selected facility.
    """

    nDems = len(r_resid)
    livings = np.where(r_resid > 0)[0]  # alive users
    unconst = np.array(list(unbuilt))  # unconstructed facilities

    # get relevant sub-arrays
    A = A_resid[np.ix_(livings, unconst)]
    f = f_fixed[np.ix_(unconst)]

    # compute contributions by as column sums
    contributions = A.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_costs = f / contributions  # unit-cost of contributions

    # select minimum costs facility
    best_fac = np.argmin(unit_costs)  # select lowest cost facility
    facility = unconst[best_fac]  # global index of selected facility

    # update residual instance
    reqs_new = np.maximum(r_resid - A_resid[:, facility], 0)
    a_max = reqs_new.reshape(nDems, 1)  # make broadcastable with A
    A_new = np.minimum(A_resid, a_max)
    unbuilt_new = unbuilt - set([facility])

    return reqs_new, A_new, unbuilt_new, facility


def greedy(r, A, f, eval_factor=False):
    """
    Run greedy algorithm for CIPs.

    Parameters
    ----------
    r : array_like
        (nDems, ) vector of requirements for each demand.
    A : array_like
        (nDems, nFacs) matrix of contributions.
    f : array_like
        (n_facs, ) vector of fixed facility costs.
    eval_factor : bool
        Set to ``True`` to evaluate APX-factor.

    Returns
    -------
    out : dict
        Output dictionary containing ``cost`` and constructed ``facilities``.
    """

    nDems, nFacs = len(r), len(f)
    nFacs = len(f)

    if A.shape[0] != nDems:
        raise ValueError("Dimension 0 of `A` and len of `r` do not match.")
    elif A.shape[1] != nFacs:
        raise ValueError("Dimension 1 of `A` and len of `f` do not match.")

    # residual values
    r_res = r
    A_res = A
    unbuilt = set(np.arange(len(f)))

    # shave off excess contributions of A
    a_max = r_res.reshape(nDems, 1)
    A_res = np.minimum(A_res, a_max)

    # standardize instance
    A_res = A_res / r_res[:, np.newaxis]  # A_res in [0,1]
    r_res = np.repeat(1, r_res.shape)

    construct = []
    residuals = []
    # store old columnsum for optional factor evaluation
    oldcolsum = A_res.sum(axis=0)  # .copy()
    factors = []

    # main loop
    while np.any(r_res > 0):
        if not bool(unbuilt):  # if no unbuilt facilities
            return np.NaN, construct, residuals
        else:
            r_res, A_res, unbuilt, fac = _greedy_update(r_res, A_res, f, unbuilt)
            construct.append(fac)
            residuals.append(r_res)

            if eval_factor:
                # compute new column sum and factors
                newcolsum = A_res.sum(axis=0)

                with np.errstate(divide="ignore", invalid="ignore"):
                    factor = (oldcolsum - newcolsum) / oldcolsum

                # replace NaNs (0/0) and Infs (a/0, a not 0) with 0s
                factor[np.isnan(factor)] = 0
                factor[np.isinf(factor)] = 0

                factors.append(factor)
                oldcolsum = newcolsum

    # compute cost
    cost = 0
    for fac in construct:
        cost = cost + f[fac]

    if eval_factor:
        # compute max factor and return
        max_fac = sum(factors).max()
        return {
            "cost": cost,
            "facilities": construct,
            "residuals": residuals,
            "factor": max_fac,
        }
    else:
        return {"cost": cost, "facilities": construct}
