#!/usr/bin/env python

r"""
Implements determinantal thinning models.

A *determinantal thinning model* is a model for subset selection that
leverages a determinantal point process. A determinantal process over
a set :math:`X` of size :math:`n` is characterized by a p.s.d symmetric
n-by-n matrix :math:`L` know as an *L-ensemble*. The likelihood of
observing a subset :math:`Y\subseteq X` is given by

.. math::
    P_L(Y) = \frac{\det(L_Y)}{\det(I + L)}


Where :math:`L_Y` is the sub-matrix corresponding to rows and columns
in :math:`Y`, and :math:`I` is the n-by-n identity matrix. Determinantal
point processes support sampling, and many other operations.

The key to having a useful determinantal thinning model is to use an
L-ensemble *kernel function* that can take data of variable sized ground
sets and return L-ensemble matrices of appropriate size. This lets the
determinantal model generate a determinantal point process over any input.
The kernel function can depend on data associated with each element
in the ground set. This module implements such kernel functions.

A useful decomposition of the L-ensemble is splitting it into a
*quality* and a *similarity* model. On an input set of size :math:`n` the
quality model produces a vector of length :math:`n` called `q`. Meanwhile
the similarity model is a psd symmetirc :math:`n \times n` matrix :math:`S`.
The L-ensemble the matrix formed by elements

.. math::
    L_{ij} = q_i S_{ij} q_j

This module implements a ``Quality`` class and a ``Similarity`` class.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import cho_factor, cho_solve

__author__ = "Sander Aarts"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Sander Aarts"
__email__ = "sea78@cornell.edu"


# =========================================================
#        An L-ensemble class
# =========================================================


class EllEnsemble:
    r"""
    An L-ensemble kernel class.

    The L-ensemble takes ``data`` of size ``(nObs, nDims)`` and outputs a psd
    symmetric matrix ``L`` of size ``(nObs, nObs)``, which induces a
    determinantal process over the index set of ``data``.

    An ``EllEnsemble`` is a composition ``Quality`` and ``Similarity`` models.
    Given ``data`` of size ``(nObs, nDims)`` the ``Quality`` model returns a
    vector `q` of size ``(nObs, )``;  the similarity model returns a symmetric
    matrix of size ``(nObs, nObs)``. The L-ensemble ``L`` is computed as

    .. math::
        L = \operatorname{Diag}(q) S \operatorname{Diag}(q)

    where :math:`\operatorname{Diag}(q)` is the diagonal matrix with :math:`q`
    on its diagonal.
    """

    def __init__(self, quality, similarity):
        """
        Initialize L-ensemble with given quality and similariy models.
        """
        self.quality = quality
        self.similarity = similarity

    def collect_parameters(self, parameters):
        """Collect parameters for quality and similarity models.

        Parameters
        ----------
        parameters : dict
            Dictionary of variableName, parameter pairs.
        """

        paramsQual, paramsSimi = [], []

        for var in self.quality.variables:
            paramsQual.append(parameters["quality"][var])

        for var in self.similarity.variables:
            paramsSimi.append(parameters["similarity"][var])

        return np.array(paramsQual), np.array(paramsSimi)

    def evaluate(self, data, parameters):
        """
        Evaluate L-enesmble.
        """
        paramsQual, paramsSimi = self.collect_parameters(parameters)

        qual = self.quality(data, paramsQual)
        simi = self.similarity(data, paramsSimi)

        return np.diag(qual) @ simi @ np.diag(qual)

    def __call__(self, data, parameters):
        return self.evaluate(data, parameters)

    def marginalKernel(self, data, parameters):
        """
        Compute the marginal kernel :math:`K`.


        The marginal kernel :math:`K` can be computed via

        .. math::
            K = L(I + L)^{-1}

        Where :math:`I` is the identity matrix and :math:`A^{-1}`
        denotes the matrix inverse.

        This implementation uses Cholesky solving for stability.

        .. math::
            K = L(I+L)^{-1} \iff (L + I)^T K^T = L^T

        We solve for :math:`K^T` using Cholesky decomposition.
        """

        L = self.evaluate(data, parameters)

        eye = np.eye(L.shape[0])

        C, low = cho_factor((L + eye).T)

        K = cho_solve((C, low), L.T).T

        return K

    def sample(self, data, parameters, seed=None):
        """
        Sample L-ensemble formed on given data and parameters.

        Parameters
        ----------
        data : dict
            Dictionary of varaibleName, array pairs.

        parameters : dict
            Dictionary of variableName, parameter pairs.

        seed : int, optional
            A seed for the RNG, also accepts generators.

        See Also
        --------
        ``determinantal_thining.dpp_sampler_generic_kernel``
        """

        K = self.marginalKernel(data, parameters)

        return dpp_sampler_generic_kernel(K, seed=seed)


# =========================================================
#        Quality models for L-ensemlbe
# =========================================================


class Quality(ABC):
    """
    Quality kernel model ABC for determinantal point processes.

    Attributes
    ----------
    variables : list of str
        List of variable names used by model.

    parameters : dict
        Dictionary of {variable : parameter} pairs.
    """

    def __init__(self, variables, parameters=None):
        self.variables = variables
        self.parameters = parameters

    def collect_data(self, data):
        """
        Collect numpy.ndarray of relevant data from broader data-dictionary.
        """
        columns = [data[var] for var in self.variables]

        return np.vstack(columns).T

    @abstractmethod
    def evaluate(self, data, parameters=None):
        """
        Evaluate the L-ensemble on given data.

        Parameters
        ----------
        data : dictionary of numpy.ndarrays
            May be richer than necesssary, but should contain self.variables.

        parameters : dict, optional
            Dictionary of {variable : parameter} pairs.
        """
        pass

    def __call__(self, data, parameters=None):
        return self.evaluate(data, parameters=parameters)


class ExponentialQuality(Quality):
    r"""
    An exponential quality model.

    The exponential quality model linearly combines variables and
    parameters and exponentiates the sum. Given a batch :math:`B` of
    :math:`n` items, where each item :math:`i` is equipped with data
    vector :math:`x_i \in \mathbb{R}^d` for some dimension :math:`d`,
    the exponential quality of item :math:`i` is given by

    .. math::
        q(x_i, \beta) = \exp\left(\frac{1}{2}\sum^d_{k=1}x_{i,k} \beta_k \right)

    It is often useful to add a constant as well, so that the sum is:

    .. math::
        \beta_0 + \sum^d_{k=1}x_{i,k} \beta_k

    This is done by simply appending a columns of :math:`1`s to the data,
    and including a variable ``constant``.

    """

    def evaluate(self, data, parameters=None):
        """
        Evaluate the quality for given item data and parameters.
        """

        data = self.collect_data(data)

        mu = data @ parameters

        q = np.exp(mu / 2)

        return q


# =========================================================
#        Similarity models for L-ensemlbe
# =========================================================


class Similarity(ABC):
    """
    Similarity kernel ABC for determinantal point processes.
    """

    def __init__(self, variables, parameters=None):
        self.variables = variables
        self.parameters = parameters

    def collect_data(self, data):
        """
        Collect numpy.ndarray of relevant data from broader data-dictionary.
        """
        columns = [data[var] for var in self.variables]

        return np.vstack(columns).T

    @abstractmethod
    def evaluate(self, data, parameters=None):
        """
        Evaluate the L-ensemble on given data.

        Parameters
        ----------
        data : dictionary of numpy.ndarrays
            May be richer than necesssary, but should contain self.variables.

        parameters : dict, optional
            Dictionary of {variable : parameter} pairs.
        """
        pass

    def __call__(self, data, parameters=None):
        return self.evaluate(data, parameters=parameters)


class RBF(Similarity):
    r"""
    A radial basis function kernel.

    The RBF is a common kernel function for symmetric psd matrices.
    Suppose there are two items :math:`i` and :math:`j` with associated
    data :math:`x_i` and :math:`x_j` in some common features space
    :math:`\mathbb{R}^d`. Suppose we have a vector of *lengthscales*
    :math:`\ell = (\ell_1, \dots, \ell_d)` mathching the feature dimension.
    Then the RBF similarity between items :math:`i` and :math:`j` is

    .. math::
        S_{RBF}(x_i, x_j \mid \ell) =
        \exp\left(-\sum^d_{k=1}\frac{|x_{i,k} - x_{j,k}|^2}{2\ell^2_k}\right)

    When the lengthscales differ by dimension the kernel is called
    _anisotrophic_. If constant lengthscale is used the kernel is _isotropic_.

    **Note:** Our implementation uses log-lengthscales :math:`\log(\ell)`.
    That is ``parameters == np.log(lengthscale)``. This is more stable for
    small lengthscales, which are common in determinantal point process
    settings.
    """

    def evaluate(self, data, parameters):
        """
        Evaluate the kenrel function at given data and parameters.
        """

        parameters = np.array(parameters)
        data = self.collect_data(data)

        lengthscale = 2 * np.exp(2 * parameters)  # 2 * ell**2

        squareDist = np.square(data[:, np.newaxis, :] - data[np.newaxis, :, :])

        S = np.exp(-np.sum(squareDist / lengthscale, axis=-1))

        return S


# =========================================================
#  Sampling algorthim(s) for Determinantal point process
# =========================================================


def dpp_sampler_generic_kernel(K, seed=None):
    r"""
    Sample from a DPP defined marginal kernel :math:`K`.

    Based on :math:`LU` factorization procedure.

    Parameters
    ----------
    K : numpy.ndarray
        A K-kernel matrix of an L-ensemble.

    seed : int, optional
        Also supports ``numpy.random._generator.Generator``-objects.

    Notes
    -----
    This implementation is based directly on the corresponing
    implementation in the `DPPy`_ package. The algorithms itself
    is is due to this `paper by Jack Poulson`_.

    .. _DPPy: https://github.com/guilgautier/DPPy
    .. _paper by Jack Poulson: https://arxiv.org/abs/1905.00165
    """

    rng = np.random.default_rng(seed=seed)

    A = K.copy()

    sample = np.repeat(False, len(A))

    uniform = rng.uniform(size=len(A))

    for j in range(len(A)):

        if uniform[j] < A[j, j]:
            sample[j] = True
        else:
            A[j, j] -= 1

        A[j + 1 :, j] /= A[j, j]
        A[j + 1 :, j + 1 :] -= np.outer(A[j + 1 :, j], A[j, j + 1 :])

    return sample
