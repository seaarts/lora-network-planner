"""
A collections of algorithms for covering integer programs.

Currenlty implemented are:
- A Greedy algorithm
- A Primal-Dual algorithm

"""

import collections.abc
import warnings

# from ortools.linear_solver import pywraplp
import numpy as np


def _dual_update(r_resid, A_resid, f_resid, unbuilt):
    """
    A dual update for the primal-dual algorithm for CIPs.
    Choose the next facility to construct out of a set
    of previously unbuilt facilities, and updates residual
    values for requirements r, contributions A, and costs f.

    ::param r_resid:: (n_dems, ) np.array of residual values
    ::param A_resid:: (n_dems, n_facs) np.array of contributions
    ::param f_resid:: (n_facs, ) np.array of item costs
    ::param unbuilt:: set of indices of unbuilt facilities
    """

    n_dems = len(r_resid)
    n_facs = len(f_resid)
    livings = np.where(r_resid > 0)[0]  # alive users
    unconst = np.array(list(unbuilt))  # unconstructed facilities

    # get relevant sub-arrays
    reqs = r_resid[np.ix_(livings)]
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


def _greedy_update(r_resid, A_resid, f_fixed, unbuilt):
    """
    A greedy step for the greedy algorithm for CIPs.
    Choose the next facility to construct out of a set
    of previously unbuilt facilities, and updates residual
    values for requirements r, contributions A. Costs
    are fixed throughout the run of the algorithm.

    ::param r_resid:: (n_dems, ) np.array of residual values
    ::param A_resid:: (n_dems, n_facs) np.array of contributions
    ::param f_fixed:: (n_facs, ) np.array of facility costs
    ::param unbuilt:: set of indices of unbuilt facilities
    """

    n_dems = len(r_resid)
    n_facs = len(f_fixed)
    livings = np.where(r_resid > 0)[0]  # alive users
    unconst = np.array(list(unbuilt))  # unconstructed facilities

    # get relevant sub-arrays
    reqs = r_resid[np.ix_(livings)]
    A = A_resid[np.ix_(livings, unconst)]
    f = f_fixed[np.ix_(unconst)]

    contributions = A.sum(axis=0)  # contribution of each facility
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_costs = f / contributions  # unit-cost of contributions

    # compute minimum costs and select best facility
    min_cost = np.min(unit_costs)  # find lowest unit cost
    best_fac = np.argmin(unit_costs)  # select lowest cost facility
    facility = unconst[best_fac]  # global index of selected facility

    # residual requirements
    reqs_new = np.maximum(r_resid - A_resid[:, facility], 0)

    a_max = reqs_new.reshape(n_dems, 1)  # make broadcastable with A
    A_new = np.minimum(A_resid, a_max)  # take min over A and residual requirements

    unbuilt_new = unbuilt - set([facility])

    return reqs_new, A_new, unbuilt_new, facility


def greedy(r, A, f, eval_factor=False):
    """
    Run greedy algorithm for CIPs.
    ::param r:: (n_dems, ) np.array of demand requirements
    ::param A:: (n_dems, n_facs) np.array of contributions
    ::param f:: (n_facs, ) np.arrray of facility costs
    ::param eval_factor:: bool for whether to evaluate apx factor.
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
    unbuilt = set(np.arange(len(f)))  # index of f = facility set

    # shave off excess contributions of A
    a_max = r_res.reshape(n_dems, 1)
    A_res = np.minimum(A_res, a_max)

    # standardize instance
    A_res = A_res / r_res[:, np.newaxis]  # A_res in [0,1]
    r_res = np.repeat(1, r_res.shape)

    construct = []
    residuals = []
    # store old columnsum for optional factor evaluation
    oldcolsum = A_res.sum(axis=0)  # .copy()
    factors = []

    ### main loop
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


def create_covering_model(r, A, f):
    """
    Store data for covering problem in or-tools friendly format.
    ::param r:: (n_dems,) np.array of demands
    ::param A:: (n_dems, n_facs) np.array of contributions
    ::param f:: (n_facs,) np.array of facility costs
    """
    n_facs = len(f)
    n_dems = len(r)

    if A.shape != (n_dems, n_facs):
        raise ValueError("Dimensions of A do not match those of r and f.")

    data = {}
    data["contributions"] = A
    data["requirements"] = r
    data["obj_coeffs"] = f
    data["num_vars"] = n_facs
    data["num_constraints"] = n_dems

    return data
