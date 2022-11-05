"""
The ``distributions``-module contains simple simulation models and utilities.

Simulation, and point processes in particular, play a central role in
studying wireless networks. Variables such as device locations in space
and transmission times in time are often modeled as `point processes
<https://en.wikipedia.org/wiki/Point_process>`_.
This package implements point processes through a ``SpatialDomain``-class.
The domain represents a finite observable windown, such as a study-area
on a map, or a finite window in time. Each domain must implements its own
method for sampling a homogeneous
`Poisson process <https://en.wikipedia.org/wiki/Poisson_point_process>`_;
more advanced processes are implemented on the ``SpatialDomain``-level.
"""


import numpy as np
from scipy.spatial.distance import pdist
from abc import ABC, abstractmethod
import collections.abc
import warnings

from scipy.stats import norm
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.metrics.pairwise import euclidean_distances




#=========================================================
#        General utilities and tools
#=========================================================


def _check_bbox(left, bottom, right, top):
    """Check if given bbox coordinates are well-defined and non-empty.
    """
    
    formatstr = "Coords should be (left, bot, right, top)."
        
    if left >= right:
        raise ValueError(f"left <= right. {formatstr}")
    if bottom >= top:
        raise ValueError(f"top <= bottom. {formatstr}")
    else:
        return((left, bottom, right, top)) 


def _check_scalar(x, varname="input"):
    """
    Check if x is a scalar.
    
    Notes
    -----
    This check is not robust against everything.
    
    """
    if isinstance(x, (collections.abc.Sequence, dict, np.ndarray)):
        raise ValueError(f"{varname} must be scalar.")
    else:
        return x




#=========================================================
#        (Spatial) Domains for Point Processes
#=========================================================


class SpatialDomain(ABC):
    """
    Spatial domain for a point process.
    
    Attributes
    ----------
    name : str, optional
    
    Notes
    -----
    Spatial domains must have an `area` 
    """ 
    
    def __init__(self, name=None):
        """Initialize a spatial domain.
        
        Parameters
        ----------
        name : str, optional
        """
        
        self.name = name
        
    @property
    @abstractmethod
    def measure(self):
        """A measure for the size of the domain.
        
        Notes
        -----
        For two-dimensional domains the area is a natural candidate.
        For time-intervals the length is preferred.
        """
        pass
    
    @abstractmethod
    def poisson(self, intensity, seed=None):
        """
        Sample homogeneous Poisson point process over Domain.
        
        Notes
        -----
        Sampling homogeneous Poisson Processes is a critical
        part of sampling more involved methods. By requiring each
        child class of SpatialDomain to support a `poisson` method
        allows us to implement other porcesses on the SpatialDomain
        level without repetition.
        """
        pass
    
    
    def lgcp(self, intensity_latent, gaussian_process, seed=None):
        """
        Sample a log gaussian cox process.
        
        Parameters
        ----------
        latent_intensity : float
            The intensity of a latent poisson process.
        gaussian_process: sklearn.gaussian_process_gpr.GaussianProcessRegressor
            A GP object from sklearn.
        seed: optional
    
        Returns 
        """
        
        rng = np.random.default_rng(seed)
    
        # verify GP is well-defined
        if not isinstance(gaussian_process, GaussianProcessRegressor):
            raise ValueError(f"The gaussian_process is not a {gp_type}")
    
        points_latent = self.poisson(intensity_latent, seed=rng)
    
        # form GP predictions at latent points
        mean_int, std_int = gaussian_process.predict(points_latent, return_std=True)
    
        log_intensities = rng.normal(mean_int, std_int)
        intensities = np.exp(log_intensities) / np.exp(log_intensities).max()
    
        # sample retained points given intensity
        uniforms = rng.uniform(0,1, size=points_latent.shape[0])
        ponits_observer = points_latent[uniforms <= intensities]
    
        return points_observed
    


class Rectangle(SpatialDomain):
    """
    A rectangular spatial domain for Point Process sampling.
    """
    
    def __init__(self, bbox, name=None):
        """Initialize a rectangle.
        """
        self.bbox = _check_bbox(bbox)
        self.name = name
        
    @property
    def width(self):
        """Width of rectangle.
        """
        (left, bottom, right, top) = self.bbox
        return right - left
    
    @property
    def height(self):
        """Height of rectangle.
        """
        (left, bottom, right, top) = self.bbox
        return top - bottom
    
    @property
    def area(self):
        """Area of the rectangle.
        """
        return self.width * self.height
    
    @property
    def measure(self):
        """Lebesque measure of rectangle (area).
        """
        return self.area
    
    
    def uniform(self, nPoints, seed=None):
        """Sample points uniformly over the rectangle.
        
        Parameters
        ----------
        nPoints: int
            Number of points to sample
        seed: optional
        """
       
        rng = np.random.default_rng(seed)
        
        nPoints = int(nPoints)
        
        if nPoints < 0:
            raise ValueError("Number of points must be non-negative.")
            
        points = rng.uniform(0, 1, size=(n_pts, 2)) # standard uniforms
        
        points[:,0] *= self.width  # scale x-coordinates
        points[:,1] *= self.height # scale y-coordinates
        
        return points
    
    
    def poisson(self, intensity, seed=None):
        """Sample a homogeneous Poisson Process over rectangle.
        
        Parameters
        ----------
        intensity: float
            The intensity of the poisson process.
        seed : optional
        """
        
        rng = np.random.default_rng(seed)
        
        n_pts = rng.poisson(self.area * intensity)
        
        return self.uniform(n_pts, seed=seed)



class TimeWindow(SpatialDomain):
    """A time interval.
    
    Attributes
    ----------
    tMin : float
        The left-hand end point of the time interval.
    tMax : float
        The right-hand end point of the time interval.
    buff : float
        Number of time units by which to buffer the interval
        in both left and right directions.
        
    """
    
    def __init__(self, tMin, tMax, buffer=0):
        """Specify a time-window by start and end points.
        
        Parameters
        ----------
        tMin : float
            The left-hand end point of the time interval.
        tMax : float
            The right-hand end point of the time interval.
        buff : float, optional
            Number of time units by which to buffer the interval
            in both left and right directions. Default 0.
            
        Raises
        ------
        ValueError
            If right hand end-point smaller than left-hand one.
        """
        
        if tMin >= tMax:
            
            raise ValueError("Init failed. tMin >= tMax.")
            
        
        self.tMin = tMin
        self.tMax = tMax
        self.buff = buffer
        
    def __repr__(self):
        return f"TimeWindow({self.tMin}, {self.tMax}, buffer={self.buff})"
    
    @property
    def length(self):
        """Length of TimeWindow including buffers.
        """
        
        return self.tMax - self.tMin + 2*self.buff
    
    @property
    def measure(self):
        """Length of TimeWindow including buffers.
        """
        return self.length
    
    def contains(self, x):
        """
        Check whether time window contains given point.
        
        Parameters
        ----------
        x : float or array_like
            Position of a point on real line.
            
        Returns
        -------
        contains : bool
            Indicates whether x in contained in TimeWindow
        """
        
        return (self.tMin <= x) * (x <= self.tMax)
    
    def uniform(self, size=None, seed=None):
        """
        Sample points uniformly over buffered time window.
        
        Parameters
        ----------
        size : int or tuple of ints, optional
            Shape of uniform random vector.
            Passed to `np.random.unfiorm()`
        seed : int, optional
            Seed passed to `np.random.default_rng()`.
            
        Returns
        -------
        u : ndarray or scalar
            
        """
        
        rng = np.random.default_rng(seed=seed)
        
        low = self.tMin - self.buff
        high = self.tMax + self.buff
        
        return rng.uniform(low=low, high=high, size=size) 
    
    def poisson(self, intensity, seed=None):
        """Sample homogeneous poisson process over window.
        
        Parameters
        ----------
        intensity : float
            Intensity or arrival rate.
        seed : optional
        """
        rng = np.random.default_rng(seed)
        
        nPoints = rng.poisson(self.measure * intensity)
        
        return self.uniform(nPoints, seed=seed)
    


    
class ArrivalProcess(ABC):
    """Stochastic arrival process on TimeWindow.
    
    Notes
    -----
    Should support richer distirbutions than Poisson, e.g. spatial models with duty cycles.
    """
    
    def __init__(self):
        """
        Instantiate arrival process.
        """
    
    @abstractmethod
    def sample(self):
        pass
    

class PoissonArrivals(ArrivalProcess):
    """Homogeneous Poisson arrival processes
    
    
    Attributes
    ----------
    rate : float
        Poisson arrival rate.
    """
    
    def __init__(self, timeWindow, rate=None):
        """Instantiate homogeneous Poisson process.
        
        Parameters
        ----------
        timeWindow : timeWindow
             Time window over which to simulate process.
        
        rate : float
             Poisson arrival rate.
             
             
        """
        super().__init__()
        self.timeWindow = timeWindow
        self.rate = rate
        
    def __repr__(self):
        return f"HomogeneousPoissonArrivals({self.timeWindow}, rate={self.rate})"
        
    def sample(self, size=None, rate=None, seed=None, stream=None):
        """
        Sample Poisson process over TimeWindow.
        
        Parameters
        ----------
        
        rate : float, optional
            Poisson arrival rate. May be `None` if stored in `ArrivalProcess.rate`.
        
        size : int, optional
            The number of samples to be taken. Limited to integer
            rather than tuples, because each sample may vary in
            size. Default 1.
            
        stream : list of np.random.generators, optional
            List of generators for the individual Process samples.
            Passed as seeds to to TimeWindow.sample(). This enables
            the use of Common Random Numbers in placing the points,
            when e.g. the arrival rates are varied over multiple
            experiments. Default `None`.
            
            
        Returns
        -------
        samples : list of ndarrays
            The length of `samples` is `size`.
            
            
        Raises
        ------
        ValueError
            If `self.rate` and `rate` are `None`.
        """
        
        rng = np.random.default_rng(seed=seed)
           
        
        if not rate and self.rate:
            rate = self.rate
        else:
            raise ValueError("""No rate given. Try passing`rate=1.0`,\
 or set a rate for the ArrivalProcess."""
                            )
       
        if size is None:
            size = 1
    
        if stream is None:
            stream = [rng for i in range(size)]  # pointers to same rng
        
        
        window = self.timeWindow
        
        mu = window.length * rate
        
        nPoints = np.random.poisson(mu, size=size)
        
        
        return [window.uniform(nPoints[i], seed=stream[i]) for i in range(size)]


    
    
#=========================================================
#        Thinning models for (marked) Point processes
#=========================================================

def ThinningModel(ABC):
    """
    An abstract thinning model.
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def thin():
        pass
    
    def __call__(self, points, *args, **kwargs):
        """Call the thinning model.
        """
        return self.thin(points, *args, **kwards)

    
class MaternThinning:
    """
    A Matérn thinning model.
    
    Matérn thinning is a hard-core thinning process for a collection of points
    in which each point is equipped with a mark. A ball (core) of fixed radius
    is drawn around each point. Points with overlapping cores are considered
    colliding. How collisions are resolved depends on the type. Each type is
    more forgiving than the last.
    #. Type I: All colliding points are thinned. This models scenarios in which
    it is impossible to observe points withing 2 radii of each other.
    #. Type II: Among colliding pairs, the one with the higher mark is retained,
    the other points is are thinned.
    #. Type III: Among colliding pairs, only points with colliding neighbors
    that are both higher marked *and* retained are thinned.
    
    Attributes
    ----------
    radius : float
        Fixed radius of interaction between points.
    kind : str, optional
        Default is '1', while '2' and '3' are also supported. See notes.
    
    Methods
    -------
    thin
        Thins a given (marked) point process.
        
    Notes
    -----
    The process is named after Bertil Matérn who published on them in 1962.
    """
    
    def __init__(self, radius, kind='1'):
        """
        Instantiate Matérn thinning model.
        """
        self.radius = radius
        self.kind = kind
    
    def thin(self, points, marks=None, radius=None, kind=None):
        """
        Applies Matérn thinning to a collection of points.
        
        Parameters
        ----------
        points : array_like
            An array of point coordinates of shape [nPoints, nDims] where
            nPoints is the number of points, and nDims the dimensionality
            of the space the points occupy.
        
        marks : array-like, optional
            Array of marks, used only if kind == '2' or '3'. Marks should
            support comparisons (>, <, =) and match `points` in 1st dim.
        
        radius : float, optional
            Optional radius that, if provided, overrides internal radius.
        
        kind : str, optional
            Optional kind-str that overrides internal kind if provided.
        
        Returns
        -------
        retained : array of bools
            An array of bools labeling each points is retained or thinned.
        """
   
    
        # sample start and end times of each point
        starts = [np.random.uniform(0, 3, n) for n in n_points]
        for i in range(len(starts)):
            starts[i].sort() # sort start times
        ends = [s + airtime for s in starts]
        times = [np.vstack((starts[i], ends[i])).T for i in range(len(ends))] # stack
        times = [t/airtime for t in times] # normalize times by airtime
    
        # compute overlap between points
        D = [pdist(s.reshape(len(s), 1), metric="euclidean") for s in starts]
        D = [squareform(d) for d in D]   # make full square matrices
        D = [(d < airtime)*1 for d in D] # 0-1 matrices of overlap
    
        overlap = [np.sum(d, axis=0) for d in D]
        overlap = [(ovlp > 1)*1 for ovlp in overlap]
    
        # compute dummy for overlapping with earlier point
        D = [d * (1 - np.tri(*d.shape, k=0)) for d in D] # takes upper triangle of collisions
        X = [np.sum(d, axis=0) for d in D] # vectors of nr. overlapping pre-arriving packets
    
        overlap_early = [(x > 0)*1 for x in X] # dummy overlapping with earlier point
    
        # collect data
        for i in range(len(X)):
            ones = np.ones(len(X[i]))
            zeros = np.zeros(len(X[i]))
            X[i] = np.vstack((zeros,
                              ones,
                              overlap[i],
                              overlap_early[i],
                              #starts[i]/airtime
                             )).T  # add 1s column to X
    
        # get Matérn type III inclusion
        for j in range(len(D)):
            n_points = D[j].shape[0]
            if n_points > 0:
                for i in range(1, len(D[j])):
                    if D[j][:,i].sum() > 0:
                        # reject i and remove all 1s in i's row
                        D[j][i, i:n_points] = np.zeros(n_points - i)
            X[j][:,0] = D[j].sum(axis=0)    

        
        incl  = [np.where(x[:,0]==0)[0] for x in X]
    
        X = [x[:, 1:] for x in X] # drop ground truth
    
        if kind == '2':
            # get inclusion from 3rd column: 'overlap_early'
            incl = [np.where(x[:,2]==0)[0] for x in X] 
    
        if kind == '1':
            # get inclusion from 2nd column: 'overlap'
            incl = [np.where(x[:,1]==0)[0] for x in X]
        
        return X, times, incl, D
    
    
#=========================================================
#        Distributions
#=========================================================
    
class Distribution(ABC):
    """
    Class for custom made probability distributions.
    
    Allows attaching parameters and a seed to the distribution instance
    so as to streamline object compositions using ``Distribution``-objects.
    
    Notes
    -----
    The ``__call__`` function is used for sampling. The parameters and
    defaults should be explicit. Each distribution uses a ``numpy.random.generator``,
    and should accept and optional ``seed`` to which a ``seed`` string or
    a ``numpy.random.Generator`` can be passed.
    
    """
    @classmethod
    def sample(self, size=None, **kwargs):
        pass
    
    def __call__(self, size=None, **kwargs):
        """Sample distribution
        """
        return self.sample(size=size, **kwargs)


class Normal(Distribution):
    """A normal distribution.
    
    Methods
    -------
    """
    
    def __init__(self, loc=0, scale=1, seed=None):
        """Instantiate normal distribution.
        """
        self.loc = loc
        self.scale = scale
        self.seed = seed
        
    def __repr__(self):
        return "NormalDistribution(%s)" % str(self.__dict__) 
        
    def sample(self, size=None, *, loc=None, scale=None, seed=None):
        """
        Sample of independent univariate normal random variables.
        
        Parameters
        ----------
        
        loc : float or array_like
            Location parameter(s)
        scale : float or array_like
            Scale parameter(s)
        size : int or tuple of ints
            Size of sample
        seed : int, optional
            Overrides Distribution's internal seed.
        
        Returns
        -------
        A normal sample of given size.
            
        See Also
        --------
        `numpy.random.normal <https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html>`_ 
        in the NumPy documentation .
        """ 
        if loc is None:
            loc = self.loc
        if scale is None:
            scale = self.scale
        
        # override self.seed if provided
        if not seed:
            seed = self.seed
        rng = np.random.default_rng(seed)
        
        return rng.normal(loc=loc, scale=scale, size=size)


class Choice(Distribution):
    """A finite discrete distribution.
    
    Methods
    -------
    """
    
    def __init__(self, a, p=None, replace=True, seed=None):
        """Instantiate finite discrete distribution.
        """
        self.a = a
        self.p = p
        self.replace = replace
        self.seed = seed
        
    def __repr__(self):
        return "DiscreteDistribution(%s)" % str(self.__dict__) 
        
    def sample(self, size=None, *, a=None, p=None, replace=True, seed=None):
        """
        Sample of independent univariate normal random variables.
        
        Parameters
        ----------
        
        a : array_like, int
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated from np.arange(a)
        p : 1d array_like, optional
            The probabilities associated with each entry in a.
            If not given, the sample assumes a uniform distribution over all entries in a.
        replace : bool
            Whether the sample is with or without replacement.
            Default is True, meaning that a value of a can be selected multiple times.
        seed : int, optional
            Overrides Distribution's internal seed.
        
        Returns
        -------
        A sample of given size.
            
        See Also
        --------
        ``numpy.random.Generator.choice`` in NumPy's `documentation <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html>`_.
        """ 
        if a is None:
            a = self.a
        if p is None:
            p = self.p
        if replace is None:
            replace = self.replace
        
        # override self.seed if provided
        if not seed:
            seed = self.seed
        rng = np.random.default_rng(seed)
        
        return rng.choice(size=size, a=a, p=p, replace=replace)
    
    
class Uniform(Distribution):
    """A uniform distribution.
    
    Methods
    -------
    """
    
    def __init__(self, low=0.0, high=1.0, seed=None):
        """Instantiate finite discrete distribution.
        """
        if low > high:
            raise ValueError("`low` cannot exceed high.")
            
        self.low = low
        self.high = high
        self.seed = seed
        
    def __repr__(self):
        return "UniformDistribution(%s)" % str(self.__dict__) 
        
    def sample(self, size=None, *, low=None, high=None, seed=None):
        """
        Sample of independent univariate normal random variables.
        
        Parameters
        ----------
        
        low : array_like, float
            Lower bounds for uniform distribution.
        high : array_like, float
            Upper bounds for uniform distribution.
        seed : int, optional
            Overrides Distribution's internal seed.
        
        Returns
        -------
        A sample of given size.
            
        See Also
        --------
        `numpy.random.Generator.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html>`_
        in NumPy's documentation.
        """ 
        if low is None:
            low = self.low
        if high is None:
            high = self.high
        
        # override self.seed if provided
        if not seed:
            seed = self.seed
        rng = np.random.default_rng(seed)
        
        return rng.uniform(size=size, low=low, high=high)