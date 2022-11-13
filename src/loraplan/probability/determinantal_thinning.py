r"""
A module of determinantal thinning models.

A determinantal thinning model is a model for subset selection that
leverages determinantal point processes. A determinantal process over
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

import numpy as np
from abc import ABC, abstractmethod


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
        L = \text{diag}(q) S \text{diag}(q)
    
    """
    
    def __init_(self):
        pass
    
    def sample(self):
        pass
    

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
    
    """
    
    def evaluate(self, data, parameters=None):
        """
        Evaluate the quality for given item data and parameters.
        """
        
        data = self.collect_data(data)
        
        mu = data @ parameters
    
        q = np.exp(mu / 2)
    
        return q


class Similarity(ABC):
    """
    Similarity kernel ABC for determinantal point processes.
    """
    
class RBF(Similarity):
    """
    A radial basis function kernel.
    """