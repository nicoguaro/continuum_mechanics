"""
Solid mechanics module
----------------------


"""
from __future__ import division, print_function
from sympy import simplify, Matrix, S, diff, symbols, pprint, sin
from continuum_mechanics.vector import grad, div, curl, lap_vec

x, y, z = symbols("x y z")


def navier_cauchy(u, lamda, mu, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Symmetric part of the gradient of a vector function A.
    
    Parameters
    ----------
    u : Matrix (3, 1), list
        Vector function to apply the Navier-Cauchy operator.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional parameter
        it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.
        
    Returns
    -------
    navier_op: Matrix (3, 3)
        Matrix with the components of the symmetric part of the gradient.
        The position (i, j) has as components diff(A[i], coords[j].
    """
    u = Matrix(u) 
    term1 = (lamda + 2*mu) * grad(div(u, coords, h_vec), coords, h_vec)
    term2 = mu * curl(curl(u, coords, h_vec), coords, h_vec)

    return simplify(term1 - term2)
