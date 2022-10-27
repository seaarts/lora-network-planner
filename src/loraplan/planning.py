from abc import ABC, abstractmethod

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
    
    
