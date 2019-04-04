"""
Solid mechanics module
----------------------


"""
from __future__ import division, print_function
from sympy import simplify, Matrix, S, symbols, eye
from continuum_mechanics.vector import (grad, div, curl, lap_vec, grad_vec,
                                        dual_tensor, sym_grad)

x, y, z = symbols("x y z")

#%% Classic elasticity
def navier_cauchy(u, params, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Navier-Cauchy operator of a vector function u.

    Parameters
    ----------
    u : Matrix (3, 1), list
        Vector function to apply the Navier-Cauchy operator.
    lamda : float
        Lamé's first parameter.
    mu : float, > 0
        Lamé's second parameter.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    navier_op : Matrix (3, 1)
        Components of the Navier-Cauchy operator applied to the
        displacement vector.
    """
    lamda, mu = params
    u = Matrix(u)
    term1 = (lamda + 2*mu) * grad(div(u, coords, h_vec), coords, h_vec)
    term2 = mu * curl(curl(u, coords, h_vec), coords, h_vec)

    return simplify(term1 - term2)


def strain_stress(strain, parameters):
    """
    Return the stress tensor for a given strain tensor
    and material properties lambda and mu.

    Parameters
    ----------
    u : Matrix (3, 3)
        Strain tensor.
    parameters : tuple
        Material parameters in the following order:

        lamda : float
            Lamé's first parameter.
        mu : float, > 0
            Lamé's second parameter.

    Returns
    -------
    disp_op : Matrix (3, 1)
        Displacement components.
    """
    mu, lamda = parameters
    strain_trace = strain.trace()
    stress = Matrix(3, 3, lambda i, j:
                    lamda*eye(3)[i, j] * strain_trace + 2*mu * strain[j, i])
    return stress

#%% Micropolar elasticity
def micropolar(u, phi, parameters, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Micropolar operator of a vector function u, as defined
    in [NOW]_.

    Parameters
    ----------
    u : Matrix (3, 1), list
        Displacement vector function to apply the micropolar operator.
    phi : Matrix (3, 1), list
        Microrrotation (pseudo)-vector function to apply the
        micropolar operator.
    parameters : tuple
        Material parameters in the following order:

        lamda : float
            Lamé's first parameter.
        mu : float, > 0
            Lamé's second parameter.
        alpha : float, > 0
            Micropolar parameter.
        beta : float
            Micropolar parameter.
        gamma : float, > 0
            Micropolar parameter.
        epsilon : float, > 0
            Micropolar parameter.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    disp_op : Matrix (3, 1)
        Displacement components.
    rot_op : Matrix (3, 1)
        Microrrotational components.

    References
    ----------
    .. [NOW] Witold Nowacki. Theory of micropolar elasticity.
        International centre for mechanical sciences,
        Courses and lectures, No. 25. Berlin: Springer, 1972.
    """
    lamda, mu, alpha, beta, gamma, epsilon = parameters
    u = Matrix(u)
    phi = Matrix(phi)
    u_op = (lamda + 2*mu) * grad(div(u, coords, h_vec), coords, h_vec) \
         - (mu - alpha) * curl(curl(u, coords, h_vec), coords, h_vec) \
         + 2*alpha*curl(phi, coords, h_vec)
    phi_op = (beta - 2*gamma) * curl(curl(phi, coords, h_vec), coords, h_vec)\
           - (gamma - epsilon) * curl(curl(phi, coords, h_vec), coords, h_vec)\
           + 2*alpha*curl(u, coords, h_vec) - 4*alpha*phi
    return simplify(u_op), simplify(phi_op)


def disp_def_micropolar(u, phi, coords, h_vec):
    """
    Compute strain measures for micropolar elasticity, as defined
    in [NOW]_.

    Parameters
    ----------
    u : Matrix (3, 1), list
        Displacement vector function to apply the micropolar operator.
    phi : Matrix (3, 1), list
        Microrrotation (pseudo)-vector function to apply the
        micropolar operator.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    strain : Matrix (3, 3)
        Strain tensor.
    curvature : Matrix (3, 3)
        Curvature tensor.

    References
    ----------
    .. [NOW] Witold Nowacki. Theory of micropolar elasticity.
        International centre for mechanical sciences,
        Courses and lectures, No. 25. Berlin: Springer, 1972.
    """
    strain = grad_vec(u, coords, h_vec) - dual_tensor(phi)
    curvature = grad_vec(phi, coords, h_vec)
    return strain, curvature


def strain_stress_micropolar(strain, curvature, constants):
    """
    Return the stress tensor for a given strain tensor
    and material properties lambda and mu
    """
    mu, lamda, alpha, beta, gamma, epsilon = constants
    strain_trace = strain.trace()
    curv_trace = curvature.trace()
    force_stress = Matrix(3, 3, lambda i, j:
                          lamda*eye(3)[i, j] * strain_trace
                          + (mu + alpha) * strain[j, i]
                          + (mu - alpha) * strain[i, j])
    couple_stress = Matrix(3, 3, lambda i, j:
        beta*eye(3)[i, j] * curv_trace + (gamma + epsilon) * curvature[j, i]
                           + (gamma - epsilon)*curvature[i, j])
    return force_stress, couple_stress


#%% C-CST elasticity
def c_cst(u, parameters, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Corrected-Couple-Stress-Theory (C-CST) elasticity operator of a
    vector function u, as presented in [CST]_.

    Parameters
    ----------
    u : Matrix (3, 1), list
        Vector function to apply the Navier-Cauchy operator.
    parameters : tuple
        Material parameters in the following order:

        lamda : float
            Lamé's first parameter.
        mu : float, > 0
            Lamé's second parameter.
        eta : float, >0
            Couple stress modulus in C-CST.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    c_cst : Matrix (3, 1)
        Components of the C-CST operator applied to the
        displacement vector

    References
    ----------

    .. [CST] Ali R. Hadhesfandiari, Gary F. Dargush.
        Couple stress theory for solids. International Journal
        for Solids and Structures, 2011, 48, 2496-2510.
    """
    lamda, mu, eta = parameters
    u = Matrix(u)
    term1 = (lamda + 2*mu) * grad(div(u, coords, h_vec), coords, h_vec)
    term2 = mu * curl(curl(u, coords, h_vec), coords, h_vec)
    term3 = eta * lap_vec(curl(curl(u, coords, h_vec), coords, h_vec),
                          coords, h_vec)
    return simplify(term1 - term2 + term3)


def disp_def_cst(u, coords, h_vec):
    """
    Compute strain measures for C-CST elasticity, as defined
    in [CST]_.

    Parameters
    ----------
    u : Matrix (3, 1), list
        Displacement vector function to apply the micropolar operator.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    strain : Matrix (3, 3)
        Strain tensor.
    curvature : Matrix (3, 3)
        Curvature tensor.

    References
    ----------
    .. [CST] Ali R. Hadhesfandiari, Gary F. Dargush.
        Couple stress theory for solids. International Journal
        for Solids and Structures, 2011, 48, 2496-2510.
    """
    strain = sym_grad(u, coords, h_vec)
    curvature = S(1)/4 * curl(curl(u, coords, h_vec), coords, h_vec)
    return strain, dual_tensor(curvature)


#%%
if __name__ == "__main__":
    import doctest
    doctest.testmod()
