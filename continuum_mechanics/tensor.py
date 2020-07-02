# -*- coding: utf-8 -*-
"""
Tensors module
--------------------

Functions to aid the manipulation of tensors.

"""
import numpy as np

#%% Fourth order
def christ_stiff(C, n):
    """Compute the Christoffel Stiffness tensor for a direction n.

    Compute the Christoffel Stiffness tensor [1]_ (Christoffel acoustic
    tensor [2]_) for an anisotropic material with stiffness tensor C,
    represented in Voigt notation, and a direction vector n.

    Parameters
    ----------
    C : (6,6) ndarray
        Stiffness tensor in Voigt notation.
    n : (3) ndarray
        Direction vector for the propagation.

    Returns
    -------
    Gamma : (3,3) ndarray
        Christoffel Stiffness


    References
    ----------
    .. [1] Auld, B. A. (1973). Acoustic fields and waves in solids
        (Vol. 1, p. 423). New York: Wiley.

    .. [2] Datta, S. K., Shah, A. H., & Chimenti, D. E. (2008). Elastic
        waves in composite media and structures: with applications to
        ultrasonic nondestructive evaluation. CRC Press, ISBN-10:
        1420053388, 336 pages.

    """
    Gamma = np.zeros((3, 3))
    Gamma[0, 0] = C[0, 0] * n[0]**2 + C[5, 5] * n[1]**2 + C[4, 4] * n[2]**2 + \
        2 * C[0, 5] * n[0] * n[1] + 2 * C[0, 4] * \
        n[0] * n[2] + 2 * C[4, 5] * n[1] * n[2]
    Gamma[0, 1] = C[0, 5] * n[0]**2 + C[1, 5] * n[1]**2 + C[3, 4] * n[2]**2 + \
        (C[0, 1] + C[5, 5]) * n[0] * n[1] + \
        (C[0, 3] + C[4, 5]) * n[0] * n[2] + \
        (C[3, 5] + C[1, 4]) * n[1] * n[2]
    Gamma[0, 2] = C[0, 4] * n[0]**2 + C[3, 5] * n[1]**2 + C[2, 4] * n[2]**2 + \
        (C[0, 3] + C[4, 5]) * n[0] * n[1] + \
        (C[0, 2] + C[4, 4]) * n[0] * n[2] + \
        (C[2, 5] + C[3, 4]) * n[1] * n[2]
    Gamma[1, 1] = C[5, 5] * n[0]**2 + C[1, 1] * n[1]**2 + C[3, 3] * n[2]**2 + \
        2 * C[1, 5] * n[0] * n[1] + 2 * C[3, 5] * \
        n[0] * n[2] + 2 * C[1, 3] * n[1] * n[2]
    Gamma[1, 2] = C[4, 5] * n[0]**2 + C[1, 3] * n[1]**2 + C[2, 3] * n[2]**2 + \
        (C[3, 5] + C[1, 4]) * n[0] * n[1] + \
        (C[2, 5] + C[3, 4]) * n[0] * n[2] + \
        (C[1, 2] + C[3, 3]) * n[1] * n[2]
    Gamma[2, 2] = C[4, 4] * n[0]**2 + C[3, 3] * n[1]**2 + C[2, 2] * n[2]**2 + \
        2 * C[3, 4] * n[0] * n[1] + 2 * C[2, 4] * \
        n[0] * n[2] + 2 * C[2, 3] * n[1] * n[2]
    Gamma[1, 0] = Gamma[0, 1]
    Gamma[2, 0] = Gamma[0, 2]
    Gamma[2, 1] = Gamma[1, 2]
    return Gamma