#!/usr/bin/env python

r"""
Implements Matérn thinning models for point processes.

*Thinning* is the selection of a subset of arbitrary collections of
points, usually applied to the realizations of point processes.
Matérn thinning is a type of deterministic thinning model. Given a
colleciton :math:`X` of points, it produces a subset :math:`Y \subseteq
X`. The key idea behind Matérn thinning is to generate balls around each
point :math:`x \in X`. If two points' balls intersect, at least one of
these two points must be thinned; in the output set :math:`Y` no balls
intersect. This is an example of a *hard-core* model, as each point's
ball represents a hard core that cannot collide with other cores.

In wireless interference settings Matérn thinning is used to model
wireless packet collisions. Points represent the mid-points of
transmissions, or *packets*, and the "balls" are intervals during
which each transmission is on-air. If two transmissions' intervals
overlap they are said to be colliding, and only one of the packes
can be received. The thinned packets are considered lost.

See Also
--------
interference, and determinantal thinnig.

"""

import collections.abc
import numpy as np
from scipy.spatial.distance import pdist, squareform

def maternThinningI(points, radius, metric='euclidean', **kwargs):
    r"""
    Apply Matérn Type I thinning to a collection of points.
    
    Matérn Type I thinning is a determinisitc thinning process for
    selecting a subset of points among a given set. Thi is a hard-core
    point process; each point represents the center of a ball with
    a hard-core. The thinned output consists of all non-overlapping 
    points. In other words, every point that are too close to any
    of the original points is thinned.
    
    Letting ``D[i, j]`` be the distance between points ``i`` and ``j``
    this function returns ``True`` for the ``i`` th entry if for all
    ``j`` if holds that ``D[i, j] > radius[i] + radius[j]``.
    
    Parameters
    ----------
    points : array_like
        An array of shape (nPoints, nDims) of points in nDims
        dimensional euclidean space.
        
    radius : float or array_like
        A radius for the cores around the points. Can be array_like
        in order to equip each point with its own radius.
    
    metric : str or function, optional
        Passed to `scipy.spatial.distance.pdist`.
    
    kwargs
        Passed to `scipy.spatial.distance.pdist`.
    
    Returns
    -------
    retained : array of booleans
        An array of boolens of length nPoints indicating whether a
        point is retained (True) or thinned (False).
    """

    if not isinstance(radius, (collections.abc.Sequence, np.ndarray)):
        radius = np.repeat(radius, points.shape[0])
    else:
        if radius.shape[0] != points.shape[0]:
            raise ValueError("Number of points and number of radii do not match.")
    
    # compute pairwise distances
    D = squareform(pdist(points, metric=metric, **kwargs))
    
    np.fill_diagonal(D, np.inf)
    
    R = radius[np.newaxis,] + radius[np.newaxis,].T
    
    return (D > R).all(axis=1)