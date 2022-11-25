"""
Implements wrappers around NumPy distributions for attaching parameters and generators.
"""

from abc import ABC

import numpy as np


class Distribution(ABC):
    """
    Wrapper for custom probability distributions.

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
        """Sample distribution"""
        return self.sample(size=size, **kwargs)


class Normal(Distribution):
    """A normal distribution.

    Methods
    -------
    """

    def __init__(self, loc=0, scale=1, seed=None):
        """Instantiate normal distribution."""
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
        """Instantiate finite discrete distribution."""
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
            If not given, the sample assumes a uniform distribution over
            all entries in a.
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
        ``numpy.random.Generator.choice`` in NumPy's
        `documentation <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html>`_.
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
        """Instantiate finite discrete distribution."""
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
