#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2021 Alexis Pietak
# See "LICENSE" for further details.


'''
**Hard-Coded Radial Basis Functions using Gaussian RBF**

'''

# ....................{ IMPORTS                           }....................
from beartype import beartype
import numpy as np
from numpy import ndarray
# from ionspire.error import IonspireGeoException

@beartype
def gauss_wr_f(rij: ndarray, h: float) -> ndarray:
    """
    Computes the base RBF on a set of distances (uses Gaussian rbf).
    Applies to 2d or 3d as only the distance (rij) is utilized.

    Parameters
    ------------
    rij : ndarray
        Distance values for input set.
    h : float
        Scale factor for the RBF.

    Returns
    ---------
    ndarray
        Computation of this RBF on the distance matrix constructed from the input points set.

    """
    val = np.exp(-rij ** 2 / h ** 2)

    return val
