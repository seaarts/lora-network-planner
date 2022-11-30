#!/usr/bin/env python

r"""
Implements spatial and temporal domains that support point processes.

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

import collections.abc
from abc import ABC, abstractmethod

import numpy as np

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.metrics.pairwise import euclidean_distances

# =========================================================
#        General utilities and tools
# =========================================================


def _check_bbox(left, bottom, right, top):
    """Check if given bbox coordinates are well-defined and non-empty."""

    formatstr = "Coords should be (left, bot, right, top)."

    if left >= right:
        raise ValueError(f"left <= right. {formatstr}")
    if bottom >= top:
        raise ValueError(f"top <= bottom. {formatstr}")
    else:
        return (left, bottom, right, top)


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


# =========================================================
#        (Spatial) Domains for Point Processes
# =========================================================


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
        # if not isinstance(gaussian_process, GaussianProcessRegressor):
        #    raise ValueError(f"The gaussian_process is not a {gp_type}")

        points_latent = self.poisson(intensity_latent, seed=rng)

        # form GP predictions at latent points
        mean_int, std_int = gaussian_process.predict(points_latent, return_std=True)

        log_intensities = rng.normal(mean_int, std_int)
        intensities = np.exp(log_intensities) / np.exp(log_intensities).max()

        # sample retained points given intensity
        uniforms = rng.uniform(0, 1, size=points_latent.shape[0])
        points_observed = points_latent[uniforms <= intensities]

        return points_observed


class Rectangle(SpatialDomain):
    """
    A rectangular spatial domain for Point Process sampling.
    """

    def __init__(self, bbox, name=None):
        """Initialize a rectangle."""
        self.bbox = _check_bbox(*bbox)
        self.name = name

    @property
    def width(self):
        """Width of rectangle."""
        (left, bottom, right, top) = self.bbox
        return right - left

    @property
    def height(self):
        """Height of rectangle."""
        (left, bottom, right, top) = self.bbox
        return top - bottom

    @property
    def area(self):
        """Area of the rectangle."""
        return self.width * self.height

    @property
    def measure(self):
        """Lebesque measure of rectangle (area)."""
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

        points = rng.uniform(0, 1, size=(nPoints, 2))  # standard uniforms

        points[:, 0] = points[:, 0] * self.width + self.bbox[0]
        points[:, 1] = points[:, 1] * self.height + self.bbox[1]

        return points

    def poisson(self, intensity, seedNum=None, seedPos=None):
        """Sample a homogeneous Poisson Process over rectangle.

        Parameters
        ----------
        intensity: float
            The intensity of the poisson process.
        seedNum : optional
            Seed used to sample number of points.
        seedPos : optional
            Seed used to sample point positions.
        """

        rngNum = np.random.default_rng(seedNum)

        nPoints = rngNum.poisson(self.area * intensity)

        return self.uniform(nPoints, seed=seedPos)


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

    def __init__(self, tMin=0, tMax=1, buffer=0):
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
        """Length of TimeWindow including buffers."""

        return self.tMax - self.tMin + 2 * self.buff

    @property
    def measure(self):
        """Length of TimeWindow including buffers."""
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

    def poisson(self, intensity=1, seedNum=None, seedPos=None):
        """Sample homogeneous poisson process over window.

        Parameters
        ----------
        intensity : float
            Intensity or arrival rate.
        seedNum : optional
            Seed used to sample number of points.
        seedPos : optional
            Seed used to sample point positions.
        """
        rngNum = np.random.default_rng(seedNum)

        nPoints = rngNum.poisson(self.measure * intensity)

        return self.uniform(nPoints, seed=seedPos)


class ArrivalProcess(ABC):
    """Stochastic arrival process on TimeWindow.

    Notes
    -----
    Should support richer distirbutions than Poisson,
    e.g. spatial models with duty cycles.
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
        elif not rate:
            raise ValueError(
                """No rate given. Try passing`rate=1.0`,\
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
