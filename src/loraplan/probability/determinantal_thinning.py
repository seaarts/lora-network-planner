"""Determinantal thinning models"""

from abc import ABC, abstractmethod

class EllEnsemble:
    """
    An L-Esnemble class.
    """
    
    def __init_(self):
        pass
    
    def sample(self):
        pass
    

class Quality(ABC):
    """
    Quality kernel model ABC for determinantal point processes.
    """
    
class Similarity(ABC):
    """
    Similarity kernel ABC for determinantal point processes.
    """
    

class ExponentialQuality(Quality):
    """
    Exponential quality class.
    """

class RBF(Similarity):
    """
    A radial basis function kernel.
    """
    
