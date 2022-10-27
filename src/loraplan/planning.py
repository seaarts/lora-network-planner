class Demands:
    """ Demand point class.
    """
    
    def __init__(self, locs, reqs):
        """ Initialize class.
        ::param locs:: location coordinates of points (numpy.array)
        ::param reqs:: requirement vector (numpy.array)
        """

        # verify locations
        locs = np.atleast_2d(locs)
        
        if reqs.ndim != 1:
            raise ValueError("Requirements should be a 1-dimensional array.")
        
        if locs.shape[0] != len(reqs):
            raise ValueError("Number of rows in locs and reqs do not match")
            
        elif locs.shape[1] != 2:
            raise ValueError("Number of columns in locs not equal to 2.")
        
        self.locs = locs
        self.reqs = reqs
        
    @property
    def n_pts(self):
        """The number of demand points.
        """
        return self.reqs.shape[0]




class Facilities:
    """ Facility class.
    """
    
    def __init__(self, locs, cost):
        """ Initialize class.
        ::param locs:: location coordinates of points (numpy.array)
        ::param cost:: requirement vector (numpy.array)
        """

        # verify locations
        locs = np.atleast_2d(locs)
        
        if cost.ndim != 1:
            raise ValueError("Cost should be a 1-dimensional array.")
        
        if locs.shape[0] != len(cost):
            raise ValueError("Number of rows in locs and reqs do not match")
            
        elif locs.shape[1] != 2:
            raise ValueError("Number of columns in locs not equal to 2.")
        
        self.locs = locs
        self.cost = cost
        
    @property
    def n_pts(self):
        """Get number of facility locations.
        """
        return self.cost.shape[0]


### Connection Quality models

class ConnectionQuality(ABC):
    """
    Abstact base class for connection quality.
    
    This is a class of functions that take demand and facility location
    arrays as inputs and return an (n_dems, n_facs) array of qualities.
    
    Quality models may be stochastic or deterministic. All stochastic
    models should have a 'seed' variable that can take an RNG.
    
    Quality models may use the path loss models below. These take
    distances in meters (m) unless stated otherwise.
    """
    
    def __init__(self, name):
        """Initialize Quality model.
        ::param name:: str, a name for the quality model.
        """
        self.name = name
        
        
    @abstractmethod
    def __call__(self, locs_dem, locs_fac, *args, **kwargs):
        """Abstact method for computing connection quality.
        """
        pass
    
    @property
    @abstractmethod
    def deterministic(self):
        """Bool, True if no randomness is used.
        """
        pass
    
    

    
class LogErrorRate(ConnectionQuality):
    """
    Negatie log error rate quality model.
    
    This class of quality model combines path loss and iid gaussian noice:
        1. demands have fixed transmitted power (p_tx)
        2. a path loss model determines path loss (PL)
        3. iid (0, sigma) gaussian noise sampled
        4. received power (p_rx) follows from the link-budget:
            
            p_rx = p_tx - PL + noise
            
        5. a transmission is received iff minimum power (p_min) exceeded
        
        The probabiliy of success is expressed analytically as
    
            P[fail] = P[p_tw - PL + noise <= p_min ]
                    = P[noise <= p_min - p_tx + PL ]
                    = normal.cdf(p_min - p_tw + PL)
    """
    
    def __init__(self, path_loss, p_tx, p_min, stdv):
        """
        Initialize LogErrorRate Quality model
        ::param path_loss:: instance of LogLinearPathLoss
        ::param p_tx:: scalar, transmitted power (dBm)
        ::param p_min:: scalar, minimum threshold power (dBm)
        ::pram stdv:: scalar >0, standard deviation of gaussian noise
        """
        
        if not issubclass(type(path_loss), LogLinearPathLoss):
            raise ValueError("'path_loss' must be subclass of 'LogLinearPathLoss'.")
            
        p_tx = check_scalar(p_tx, varname="p_tx")
        p_min = check_scalar(p_min, varname="p_min")
        stdv = check_scalar(stdv, varname="stdv")
        
        if stdv < 0:
            raise ValueError("'stdv' must be non-negative.")
            
        self.path_loss = path_loss
        self.p_tx = p_tx
        self.p_min = p_min
        self.stdv = stdv
        
        super().__init__(name='LogErrorRate')
    
    
    def __call__(self, locs_dem, locs_fac):
        """Compute negative log error rate.
        """
        
        locs_dem = np.atleast_2d(locs_dem)
        locs_fac = np.atleast_2d(locs_fac)
        
        distances = euclidean_distances(locs_dem, locs_fac)
        path_loss = self.path_loss(distances)
        
        error_rate = norm.cdf(self.p_min + path_loss - self.p_tx, scale=self.stdv)
        
        return -np.log(error_rate)

    @property
    def deterministic(self):
        """Boolean - always true. This method is deterministic.
        """
        return True
    
    
"""
Algorithms for solving covering integer programs.
"""

#from ortools.linear_solver import pywraplp
import numpy as np
import collections.abc
import warnings


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
    livings = np.where(r_resid > 0)[0] # alive users
    unconst = np.array(list(unbuilt))  # unconstructed facilities
    
    # get relevant sub-arrays 
    reqs = r_resid[np.ix_(livings)]
    A = A_resid[np.ix_(livings, unconst)]
    f = f_resid[np.ix_(unconst)]
    
    contributions = A.sum(axis=0)  # contribution of each facility
    unit_costs = f / contributions # unit-cost of contributions
    
    # compute minimum costs and select best facility 
    min_cost = np.min(unit_costs)    # find lowest unit cost
    best_fac = np.argmin(unit_costs) # select lowest cost facility
    facility = unconst[best_fac]     # global index of selected facility
    
    # residual requirements
    reqs_new = np.maximum(r_resid - A_resid[:, facility], 0)
    
    a_max = reqs_new.reshape(n_dems,1) # make broadcastable with A
    A_new = np.minimum(A_resid, a_max) # take min over A and residual requirements
    f_new = f_resid - A_resid.sum(axis=0) * min_cost # adjust remaining prices
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
    
    n_dems = len(r) # nr demands
    n_facs = len(f) # nr facilites

    if A.shape[0] != n_dems:
        raise ValueError("Dimension 0 of A and len of r do not match.")
    elif A.shape[1] != n_facs:
        raise ValueError("Dimension 1 of A and lenr of f do not match.")
    
    # initialize residual values
    r_res = r
    A_res = A
    f_res = f
    unbuilt = set(np.arange(len(f))) # index of f = facility set
    
    # shave off excess contributions of A
    a_max = r_res.reshape(n_dems, 1)
    A_res = np.minimum(A_res, a_max)
    
    construct = []
    residuals = []
    residuals.append(r)
    
    # main loop
    while np.any(r_res > 0):
        if not bool(unbuilt): # if no unbuilt facilities
            return np.NaN, construct, residuals
        else:   
            r_res, A_res, f_res, unbuilt, fac = _dual_update(r_res, A_res, f_res, unbuilt)
            construct.append(fac)
            residuals.append(r_res)   
            
    # compute cost
    cost = 0
    for fac in construct:
        cost = cost + f[fac]
    
    else:
        return {'cost': cost, 
                'facilities': construct,
                'residuals': residuals}




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
    livings = np.where(r_resid > 0)[0] # alive users
    unconst = np.array(list(unbuilt))  # unconstructed facilities
    
    # get relevant sub-arrays 
    reqs = r_resid[np.ix_(livings)]
    A = A_resid[np.ix_(livings, unconst)]
    f = f_fixed[np.ix_(unconst)]
    
    contributions = A.sum(axis=0)  # contribution of each facility
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_costs = f / contributions # unit-cost of contributions
    
    # compute minimum costs and select best facility 
    min_cost = np.min(unit_costs)    # find lowest unit cost
    best_fac = np.argmin(unit_costs) # select lowest cost facility
    facility = unconst[best_fac]     # global index of selected facility
    
    # residual requirements
    reqs_new = np.maximum(r_resid - A_resid[:, facility], 0)
    
    a_max = reqs_new.reshape(n_dems,1) # make broadcastable with A
    A_new = np.minimum(A_resid, a_max) # take min over A and residual requirements
    
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
    
    n_dems = len(r) # nr demands
    n_facs = len(f) # nr facilites

    if A.shape[0] != n_dems:
        raise ValueError("Dimension 0 of A and len of r do not match.")
    elif A.shape[1] != n_facs:
        raise ValueError("Dimension 1 of A and lenr of f do not match.")
    
    # initialize residual values
    r_res = r
    A_res = A
    unbuilt = set(np.arange(len(f))) # index of f = facility set
    
    # shave off excess contributions of A
    a_max = r_res.reshape(n_dems, 1)
    A_res = np.minimum(A_res, a_max)
    
    # standardize instance
    A_res = A_res / r_res[:,np.newaxis] # A_res in [0,1]
    r_res = np.repeat(1, r_res.shape)
    
    construct = []
    residuals = []
    # store old columnsum for optional factor evaluation
    oldcolsum = A_res.sum(axis=0) #.copy()
    factors = []
    
    ### main loop
    while np.any(r_res > 0):
        if not bool(unbuilt): # if no unbuilt facilities
            return np.NaN, construct, residuals
        else:   
            r_res, A_res, unbuilt, fac = _greedy_update(r_res, A_res, f, unbuilt)
            construct.append(fac)
            residuals.append(r_res)
            
            if eval_factor:
                # compute new column sum and factors
                newcolsum = A_res.sum(axis=0)
                
                with np.errstate(divide='ignore', invalid='ignore'):
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
        return {'cost': cost,
                'facilities': construct,
                'residuals': residuals,
                'factor': max_fac}
    else:
        return {'cost': cost,
                'facilities': construct}
    

    
    
    
    
    
    
    
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
    data['contributions'] = A
    data['requirements'] = r
    data['obj_coeffs'] = f
    data['num_vars'] = n_facs
    data['num_constraints'] = n_dems
    
    return data