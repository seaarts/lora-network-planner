"""
Impelements a Primal-Dual / Dual Ascent algorithm for CIPs.
"""

import numpy as np


def _dual_update(r_resid, A_resid, f_resid, unbuilt):
    """
    A dual update for the primal-dual algorithm for CIPs.
    Choose the next facility to construct out of a set
    of previously unbuilt facilities, and updates residual
    values for requirements r, contributions A, and costs f.

    Prameters
    ---------
    r_resid : array_like
        A ``(nDems, )`` array of residual values.
    A_resid : array_like
        A ``(nDems, nFacs)`` array of contributions.
    f_resid : array_like
        A ``(nFacs)`` array of residual costs.
    unbuilt : set
        Available indices of unbuilt facilities.
    """

    n_dems = len(r_resid)
    livings = np.where(r_resid > 0)[0]  # alive users
    unconst = np.array(list(unbuilt))  # unconstructed facilities

    # get relevant sub-arrays
    A = A_resid[np.ix_(livings, unconst)]
    f = f_resid[np.ix_(unconst)]

    contributions = A.sum(axis=0)  # contribution of each facility
    unit_costs = f / contributions  # unit-cost of contributions

    # compute minimum costs and select best facility
    min_cost = np.min(unit_costs)  # find lowest unit cost
    best_fac = np.argmin(unit_costs)  # select lowest cost facility
    facility = unconst[best_fac]  # global index of selected facility

    # residual requirements
    reqs_new = np.maximum(r_resid - A_resid[:, facility], 0)

    a_max = reqs_new.reshape(n_dems, 1)  # make broadcastable with A
    A_new = np.minimum(A_resid, a_max)  # take min over A and residual requirements
    f_new = f_resid - A_resid.sum(axis=0) * min_cost  # adjust remaining prices
    # note: dead users have A_resid == 0 and so do not contribute

    unbuilt_new = unbuilt - set([facility])

    return reqs_new, A_new, f_new, unbuilt_new, facility


def primal_dual(r, A, f):
    """
    Run primal-dual algorithm for CIPs.
    ::param r:: len n_dems array of demand requirements
    ::param A:: (n_dems, n_facs) array of contributions
    ::param f:: len n_facs arrray of facility costs
    """

    n_dems = len(r)  # nr demands
    n_facs = len(f)  # nr facilites

    if A.shape[0] != n_dems:
        raise ValueError("Dimension 0 of A and len of r do not match.")
    elif A.shape[1] != n_facs:
        raise ValueError("Dimension 1 of A and lenr of f do not match.")

    # initialize residual values
    r_res = r
    A_res = A
    f_res = f
    unbuilt = set(np.arange(len(f)))  # index of f = facility set

    # shave off excess contributions of A
    a_max = r_res.reshape(n_dems, 1)
    A_res = np.minimum(A_res, a_max)

    construct = []
    residuals = []
    residuals.append(r)

    # main loop
    while np.any(r_res > 0):
        if not bool(unbuilt):  # if no unbuilt facilities
            return np.NaN, construct, residuals
        else:
            r_res, A_res, f_res, unbuilt, fac = _dual_update(
                r_res, A_res, f_res, unbuilt
            )
            construct.append(fac)
            residuals.append(r_res)

    # compute cost
    cost = 0
    for fac in construct:
        cost = cost + f[fac]

    else:
        return {"cost": cost, "facilities": construct, "residuals": residuals}
