import numpy as np
from abc import ABC, abstractmethod

#=========================================================
#        Simulation tools and utilities
#=========================================================


class TimeWindow():
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
        """
        Length of TimeWindow including buffers.
        """
        
        return self.tMax - self.tMin + 2*self.buff
    
    
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
    
    
class ArrivalProcess(ABC):
    """Stochastic arrival process on TimeWindow"""
    
    def __init__(self):
        """
        Instantiate arrival process.
        """
    
    @abstractmethod
    def sample(self):
        pass
    

class PoissonProcess(ArrivalProcess):
    """Homogeneous Poisson arrival processes
    
    
    Attributes
    ----------
    rate : float
        Poisson arrival rate.
    """
    
    def __init__(self, TimeWindow, rate=None):
        """Instantiate homogeneous Poisson process.
        
        Parameters
        ----------
        timeWindow : timeWindow
             Time window over which to simulate process.
        
        rate : float
             Poisson arrival rate.
             
             
        """
        super().__init__()
        self.timeWindow = TimeWindow
        self.rate = rate
        
    def __repr__(self):
        return f"HomogeneousPoissonProcess({self.timeWindow}, rate={self.rate})"
        
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