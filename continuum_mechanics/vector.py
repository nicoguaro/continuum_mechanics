"""
Vector calculus module
----------------------


"""
from __future__ import division, print_function
from sympy import simplify, Matrix, S, diff, symbols, zeros, eye
from sympy import sin, sinh, cos, cosh, sqrt

x, y, z = symbols("x y z")


#%% Curvilinear coordinates
def transform_coords(coord_sys, coords, a=1, b=1, c=1):
    """
    Return transformation for predefined coordinate systems.

    Parameters
    -------
    coord_sys : string
        Coordinate system.
    coords : Tuple (3)
        Coordinates for the new reference system.
    a : SymPy expression, optional
        Additional parameter for some coordinate systems.
    b : SymPy expression, optional
        Additional parameter for some coordinate systems.
    c : SymPy expression, optional
        Additional parameter for some coordinate systems.

    Returns
    -------
    h_vec : Tuple (3)
        Scale coefficients.

    References
    ----------
    .. [ORTHO] Wikipedia contributors, 'Orthogonal coordinates',
        Wikipedia, The Free Encyclopedia, 2019
    """
    if not isinstance(coord_sys, str):
        raise TypeError("The coordinate system should be defined by a string")
    u, v, w = coords
    r_dict = {
        "cartesian":
            (u, v, w),
        "cylindrical":
            (u*cos(v), u*sin(v), w),
        "spherical":
            (u*sin(v)*cos(w), u*sin(v)*sin(w), u*sin(v)),
        "parabolic_cylindrical":
            ((u**2 - v**2)/2, u*v, w),
        "parabolic":
            (u*v*cos(w), u*v*sin(w), (u**2 - v**2)/2),
        "paraboloidal":
            (sqrt((a**2 - u)*(a**2 - v)*(a**2 - w)/(b**2 - a**2)),
             sqrt((b**2 - u)*(b**2 - v)*(b**2 - w)/(a**2 - b**2)),
             S(1)/2*(a**2 + b**2 - u - v - w)),
        "elliptic_cylindrical":
            (a*cosh(u)*cos(v), a*sinh(u)*sin(v), w),
        "oblate_spheroidal":
            (a*cosh(u)*cos(v)*cos(w),
             a*cosh(u)*cos(v)*sin(w),
             a*sinh(u)*sin(v)),
        "prolate_spheroidal":
            (a*sinh(u)*sin(v)*cos(w),
             a*sinh(u)*sin(v)*sin(w),
             a*cosh(u)*cos(v)),
        "ellipsoidal":
            (sqrt((a**2 + u)*(a**2 + v)*(a**2 + w)/((a**2 - b**2)*(a**2 - c**2))),
             sqrt((b**2 + u)*(b**2 + v)*(b**2 + w)/((b**2 - a**2)*(b**2 - c**2))),
             sqrt((c**2 + u)*(c**2 + v)*(c**2 + w)/((c**2 - b**2)*(c**2 - a**2)))),
        "bipolar_cylindrical":
            (a*sinh(v)/(cosh(v) - cos(u)), a*sin(u)/(cosh(v) - cos(u)), w),
        "toroidal":
            (a*sinh(v)*cos(w)/(cosh(v) - cos(u)),
             a*sinh(v)*sin(w)/(cosh(v) - cos(u)),
             a*sin(u)/(cosh(v) - cos(u))),
        "bispherical":
            (a*sin(u)*cos(w)/(cosh(v) - cos(u)),
             a*sin(u)*sin(w)/(cosh(v) - cos(u)),
             a*sinh(v)/(cosh(v) - cos(u))),
        "conical":
            (u*v*w/(a*b),
             u/a*sqrt((v**2 - a**2)*(w**2 - a**2)/(a**2 - b**2)),
             u/b*sqrt((v**2 - b**2)*(w**2 - b**2)/(a**2 - b**2)))}
    if coord_sys not in r_dict.keys():
        msg = "System coordinate not available.\n\nAvailable options are:\n"
        raise ValueError(msg + ", ".join(r_dict.keys()))
    return r_dict[coord_sys]


def scale_coeff(r_vec, coords):
    """
    Compute scale coefficients for the vector
    tranform given by r_vec.

    Parameters
    -------
    r_vec : Matrix (3, 1)
        Transform vector (x, y, z) as a function of coordinates
        u1, u2, u3.
    coords : Tuple (3)
        Coordinates for the new reference system.

    Returns
    -------
    h_vec : Tuple (3)
        Scale coefficients.
    """
    if isinstance(r_vec, list):
        r_vec = Matrix(r_vec)
    u1, u2, u3 = coords
    h1 = simplify((r_vec.diff(u1)).norm())
    h2 = simplify((r_vec.diff(u2)).norm())
    h3 = simplify((r_vec.diff(u3)).norm())
    return h1, h2, h3


def scale_coeff_coords(coord_sys, coords, a=1, b=1, c=1):
    """
    Return scale factors for predefined coordinate system.

    Parameters
    -------
    coord_sys : string
        Coordinate system.
    coords : Tuple (3)
        Coordinates for the new reference system.
    a : SymPy expression, optional
        Additional parameter for some coordinate systems.
    b : SymPy expression, optional
        Additional parameter for some coordinate systems.
    c : SymPy expression, optional
        Additional parameter for some coordinate systems.

    Returns
    -------
    h_vec : Tuple (3)
        Scale coefficients.

    References
    ----------
    .. [ORTHO] Wikipedia contributors, 'Orthogonal coordinates',
        Wikipedia, The Free Encyclopedia, 2019
    """
    if not isinstance(coord_sys, str):
        raise TypeError("The coordinate system should be defined by a string")
    u, v, w = coords
    h_dict = {
        "cartesian":
            (1, 1, 1),
        "cylindrical":
            (1, u, 1),
        "spherical":
            (1, u, u*sin(v)),
        "parabolic_cylindrical":
            (sqrt(u**2 + v**2), sqrt(u**2 + v**2), 1),
        "parabolic":
            (sqrt(u**2 + v**2), sqrt(u**2 + v**2), u*v),
        "paraboloidal":
            (S(1)/2*sqrt((v - u)*(w - u))/((a**2 - u)*(b**2 - u)),
             S(1)/2*sqrt((w - v)*(u - v))/((a**2 - v)*(b**2 - v)),
             S(1)/2*sqrt((u - w)*(v - w))/((a**2 - w)*(b**2 - w))),
        "elliptic_cylindrical":
            (a*sqrt(sinh(u)**2 + sin(v)**2),
             a*sqrt(sinh(u)**2 + sin(v)**2), 1),
        "oblate_spheroidal":
            (a*sqrt(sinh(u)**2 + sin(v)**2),
             a*sqrt(sinh(u)**2 + sin(v)**2),
             a*sinh(u)*sin(v)),
        "prolate_spheroidal":
            (a*sqrt(sinh(u)**2 + sin(v)**2),
             a*sqrt(sinh(u)**2 + sin(v)**2),
             a*cosh(u)*cos(v)),
        "ellipsoidal":
            (S(1)/2*sqrt((v - u)*(w - u))/((a**2 - u)*(b**2 - u)*(c**2 - u)),
             S(1)/2*sqrt((w - v)*(u - v))/((a**2 - v)*(b**2 - v)*(c**2 - v)),
             S(1)/2*sqrt((u - w)*(v - w))/((a**2 - w)*(b**2 - w)*(c**2 - w))),
        "bipolar_cylindrical":
            (a/(cosh(v) - cos(u)), a/(cosh(v) - cos(u)), 1),
        "toroidal":
            (a/(cosh(v) - cos(u)), a/(cosh(v) - cos(u)),
             a*sinh(v)/(cosh(v) - cos(u))),
        "bispherical":
            (a/(cosh(v) - cos(u)), a/(cosh(v) - cos(u)),
             a*sin(v)/(cosh(v) - cos(u))),
        "conical":
            (1,
             u*sqrt((v**2 - w**2)/((v**2 - a**2)*(b**2 - b**2))),
             u*sqrt((v**2 - w**2)/((v**2 - a**2)*(b**2 - b**2))))}
    if coord_sys not in h_dict.keys():
        msg = "System coordinate not available.\n\nAvailable options are:\n"
        raise ValueError(msg + ", ".join(h_dict.keys()))
    return h_dict[coord_sys]


def unit_vec_deriv(vec_i, coord_j, coords=(x, y, z), h_vec=(1, 1, 1)):
    r"""
    Compute the derivatives of unit vectors with respect
    to coordinates

    The derivative is defined as

    .. math::

        \frac{\partial{\hat{\mathbf{e}}_i}}{\partial u_j} =
        \begin{cases}
        \hat{\mathbf{e}_j} \frac{1}{h_i} \frac{\partial{h_j}}{\partial u_i}
          &\text{if } i\neq j\\
        -\sum_{\substack{k=1\\ k\neq i}}^3 \hat{\mathbf{e}_k} \frac{1}{h_k}
           \frac{\partial{h_i}}{\partial u_k} &\text{if } i = j\\
        \end{cases}\, ,

    as presented in [ARFKEN]_.

    Parameters
    ----------
    vec_i : int
        Number of the unit vector (0, 1, 2).
    coord_j : int
        Number of the coordinate (0, 1, 2).
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    deriv: Matrix (3, 1)
        Derivative of the i-th unit vector with respect to the
        j-th coordinate.

    References
    ----------

    .. [ARFKEN] George Arfken, Hans J. Weber and Frank Harris.
        Mathematical methods for physicists, Elsevier, 2013.

    """
    deriv = zeros(3, 1)
    if vec_i != coord_j:
        deriv[coord_j] = diff(h_vec[coord_j], coords[vec_i])/h_vec[vec_i]
    else:
        for cont in range(3):
            if cont != vec_i:
                deriv[cont] = -diff(h_vec[vec_i], coords[cont])/h_vec[cont]
    return deriv


#%% Vector analysis
def levi_civita(i, j, k):
    """Levi-Civita symbol"""
    return (i - j)*(j - k)*(k - i)/S(2)


def dual_tensor(vec):
    r"""Compute the dual tensor for an axial vector

    In index notation, the dual is defined by

    .. math::

        C_{ij} = \epsilon_{ijk} C_k

    where :math:`\epsilon_{ijk}` is the Levi-Civita symbol.


    Parameters
    ----------
    vec : Matrix (3)
        Axial vector.

    Returns
    -------
    dual: Matrix (3, 3)
        Second order matrix that is dual of vec.

    References
    ----------

    .. [ARFKEN] George Arfken, Hans J. Weber and Frank Harris.
        Mathematical methods for physicists, Elsevier, 2013.

    """
    dual = zeros(3, 3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                dual[i, j] = dual[i, j] + levi_civita(i, j, k) * vec[k]
    return dual


def dual_vector(tensor):
    r"""Compute the dual (axial) vector for an anti-symmetric tensor

    In index notation, the dual is defined by

    .. math::

        C_{i} = \frac{1}{2}\epsilon_{ijk} C_{jk}

    where :math:`\epsilon_{ijk}` is the Levi-Civita symbol.

    Parameters
    ----------
    tensor : Matrix (3, 3)
        Second order tensor.

    Returns
    -------
    dual: Matrix (3)
        Axial vector that is the dual of tensor.

    References
    ----------

    .. [ARFKEN] George Arfken, Hans J. Weber and Frank Harris.
        Mathematical methods for physicists, Elsevier, 2013.

    """
    if not tensor.is_anti_symmetric():
        raise TypeError("The tensor should be antisymmetric")
    dual = Matrix([0, 0, 0])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                dual[i] = dual[i] + levi_civita(i, j, k) * tensor[j, k]
    return dual/S(2)


#%% Differential operators
def grad(u, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Compute the gradient of a scalara function phi.

    Parameters
    ----------
    u : SymPy expression
        Scalar function to compute the gradient from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameters, and it takes a cartesian (x, y, z), as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    gradient: Matrix (3, 1)
        Column vector with the components of the gradient.
    """
    return Matrix(3, 1, lambda i, j: u.diff(coords[i])/h_vec[j])


def grad_vec(A, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Gradient of a vector function A.

    Parameters
    ----------
    A : Matrix (3, 1), list
        Vector function to compute the gradient from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    gradient: Matrix (3, 3)
        Matrix with the components of the gradient. The position (i, j)
        has as components diff(A[i], coords[j].
    """
    gradient = zeros(3, 3)
    for i in range(3):
        vec_i = eye(3)[:, i]
        for j in range(3):
            vec_j = eye(3)[:, j]
            diff_vec = unit_vec_deriv(j, i, coords, h_vec)
            gradient += vec_i * vec_j.T * A[j].diff(coords[i])/h_vec[i]
            gradient += vec_i * diff_vec.T * A[j]/h_vec[i]
    return gradient


def sym_grad(A, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Symmetric part of the gradient of a vector function A.

    Parameters
    ----------
    A : Matrix (3, 1), list
        Vector function to compute the gradient from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    sym_grad: Matrix (3, 3)
        Matrix with the components of the symmetric part of the gradient.
        The position (i, j) has as components diff(A[i], coords[j].
    """
    G = grad_vec(A, coords, h_vec)
    return S(1)/2*(G + G.T)


def antisym_grad(A, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Antisymmetric part of the gradient of a vector function A.

    Parameters
    ----------
    A : Matrix (3, 1), list
        Vector function to compute the gradient from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    antisym_grad: Matrix (3, 3)
        Matrix with the components of the antisymmetric part of
        the gradient. The position (i, j) has as components
        diff(A[i], coords[j].
    """
    G = grad_vec(A, coords, h_vec)
    return S(1)/2*(G - G.T)


def div(A, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Divergence of the vector function A.

    Parameters
    ----------
    A : Matrix, list
        Vector function to compute the divergence from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    divergence: SymPy expression
        Divergence of A.
    """
    h = h_vec[0]*h_vec[1]*h_vec[2]
    aux = simplify((S(1)/h)*sum(diff(A[k]*h/h_vec[k], coords[k])
                                for k in range(3)))
    return aux

def div_tensor(tensor, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Divergence of a (second order) tensor

    Parameters
    ----------
    tensor : Matrix (3, 3)
        Tensor function function to compute the divergence from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    divergence: Matrix
        Divergence of tensor.

    References
    ----------
    .. [RICHARDS] Rowland Richards. Principles of Solids Mechanics.
        CRC Press, 2011.
    """
    h1, h2, h3 = h_vec
    u1, u2, u3 = coords
    div1 = diff(h2*h3*tensor[0, 0], u1) + diff(h1*h3*tensor[0, 1], u2) \
         + diff(h1*h2*tensor[0, 2], u3) + h3*tensor[0, 1]*diff(h1, u2) \
         + h2*tensor[0, 2]*diff(h1, u3) - h3*tensor[1, 1]*diff(h2, u1) \
         - h2*tensor[2, 2]*diff(h3, u1)
    div2 = diff(h2*h3*tensor[1, 0], u1) + diff(h1*h3*tensor[1, 1], u2) \
         + diff(h1*h2*tensor[1, 2], u3) + h1*tensor[1, 2]*diff(h2, u3) \
         + h3*tensor[1, 0]*diff(h2, u1) - h1*tensor[2, 2]*diff(h3, u2) \
         - h3*tensor[2, 2]*diff(h1, u2)
    div3 = diff(h2*h3*tensor[2, 0], u1) + diff(h1*h3*tensor[2, 1], u2) \
         + diff(h1*h2*tensor[2, 2], u3) + h2*tensor[2, 0]*diff(h1, u1) \
         + h1*tensor[2, 1]*diff(h1, u2) - h1*tensor[1, 1]*diff(h2, u3) \
         + h2*tensor[2, 2]*diff(h1, u3)
    return Matrix([div1, div2, div3])/(h1*h2*h3)


def curl(A, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Curl of a function vector A.

    Parameters
    ----------
    A : Matrix, List
        Vector function to compute the curl from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    curl : Matrix (3, 1)
        Column vector with the curl of A.
    """
    perm = lambda i, j, k: (i - j)*(j - k)*(k - i)/S(2)
    h = h_vec[0]*h_vec[1]*h_vec[2]
    aux = [(S(1)/h)*sum(perm(i, j, k)*h_vec[i]*diff(A[k]*h_vec[k], coords[j])
           for j in range(3) for k in range(3))
           for i in range(3)]
    return Matrix(aux)


def lap(u, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Laplacian of the scalar function u.

    Parameters
    ----------
    u : SymPy expression
        Scalar function to compute the laplacian from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameters, and it takes a cartesian (x, y, z), as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    laplacian: Sympy expression
        Laplacian of u.
    """
    h = S(h_vec[0]*h_vec[1]*h_vec[2])
    return sum([1/h*diff(h/h_vec[k]**2*u.diff(coords[k]), coords[k])
                for k in range(3)])


def lap_vec(A, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Laplacian of a vector function A.

    Parameters
    ----------
    A : Matrix, List
        Vector function to compute the laplacian from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameter it takes (x, y, z) as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    laplacian : Matrix (3, 1)
        Column vector with the components of the Laplacian.
    """
    return grad(div(A, coords, h_vec), coords, h_vec) \
           - curl(curl(A, coords, h_vec), coords, h_vec)


def biharmonic(u, coords=(x, y, z), h_vec=(1, 1, 1)):
    """
    Bilaplacian of the scalar function u.

    Parameters
    ----------
    u : SymPy expression
        Scalar function to compute the bilaplacian from.
    coords : Tuple (3), optional
        Coordinates for the new reference system. This is an optional
        parameters, and it takes a cartesian (x, y, z), as default.
    h_vec : Tuple (3), optional
        Scale coefficients for the new coordinate system. It takes
        (1, 1, 1), as default.

    Returns
    -------
    bilaplacian: Sympy expression
        Bilaplacian of u.
    """
    return lap(lap(u, coords, h_vec), coords, h_vec)


#%%
if __name__ == "__main__":
    import doctest
    doctest.testmod()
