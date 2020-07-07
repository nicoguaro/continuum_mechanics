# -*- coding: utf-8 -*-
"""
Tensors module
--------------------

Functions to aid the manipulation of tensors.

"""
import numpy as np
import sympy as sym


#%% Material properties
def iso_stiff(E, nu):
    r"""Form the stiffness tensor in Voigt notation.

    Parameters
    ----------
    E : SymPy expression
        Young modulus.
    nu : SymPy expression
        Poisson's ratio.

    Returns
    -------
    stiff : Matrix (6,6)
        Stiffness tensor in Voigt notation.

    """
    mu = E/(2*(1 + nu))
    lam = E*nu/((1 + nu)*(1 - 2*nu))
    stiff = sym.Matrix([
        [lam + 2*mu, lam, lam, 0, 0, 0],
        [lam, lam + 2*mu, lam, 0, 0, 0],
        [lam, lam, lam + 2*mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]])
    return stiff


def iso_compl(E, nu):
    """
    Form the compliance tensor in Voigt notation.

    Parameters
    ----------
    E : SymPy expression
        Young modulus.
    nu : SymPy expression
        Poisson's ratio.

    Returns
    -------
    compl : Matrix (6,6)
        Compliance tensor in Voigt notation.

    """
    compl = sym.Matrix([
        [1, -nu, -nu, 0, 0, 0],
        [-nu, 1, -nu, 0, 0, 0],
        [-nu, -nu, 1, 0, 0, 0],
        [0, 0, 0, 2*(1 + nu), 0, 0],
        [0, 0, 0, 0, 2*(1 + nu), 0],
        [0, 0, 0, 0, 0, 2*(1 + nu)]])
    return compl/E


#%% Voigt notation
def mat2voigt(tensor, case="stress"):
    """Convert from matrix notation to Voigt's notation.

    The main idea of having different representations for this two
    tensor is to keep the energy density functional as the usual
    product U = S:E, where S is the stress tensor and E is the strain
    tensor.

    Parameters
    ----------
    tensor : Matrix (3, 3)
        Second rank tensor in matrix notation.
    case : str
        Say if the tensor is written as strain or tress tensor.

    Returns
    -------
    vec : Matrix (6, 1)
        Vector-like representation of the tensor.

    References
    ----------
    .. [1] Voigt notation. In Wikipedia, The Free Encyclopedia.
        Retrieved from
        http://en.wikipedia.org/w/index.php?title=Voigt_notation

    .. [2] Stephan Puchegger (2005): Matrix and tensor notation in the
        theory of elasticity.

    .. [3] P. Helnwein (2001). Some Remarks on the Compressed Matrix
       Representation of Symmetric Second-Order and Fourth-Order Tensors.
       Computer Methods in Applied Mechanics and Engineering,
       190(22-23):2753-2770
    """
    if case == "stress":
        vec = sym.Matrix([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                      tensor[1, 2], tensor[0, 2], tensor[0, 1]])
    elif case == "strain":
       vec = sym.Matrix([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                     2*tensor[1, 2], 2*tensor[0, 2], 2*tensor[0, 1]])
    else:
        msg = "Case not supported.\n\n"\
            + "Available options are: 'stress' or 'strain'."
        raise ValueError(msg)
    return vec


def voigt2mat(vec, case="stress"):
    """Convert from matrix notation to Voigt's notation.

    The main idea of having different representations for this two
    tensor is to keep the energy density functional as the usual
    product U = S:E, where S is the stress tensor and E is the strain
    tensor.

    Parameters
    ----------
    vec : Matrix (6, 1)
        Vector-like representation of the tensor.
    case : str
        Say if the tensor is written as strain or tress tensor.

    Returns
    -------
    tensor : Matrix (3, 3)
        Second rank tensor in matrix notation.

    References
    ----------
    .. [1] Voigt notation. In Wikipedia, The Free Encyclopedia.
        Retrieved from
        http://en.wikipedia.org/w/index.php?title=Voigt_notation

    .. [2] Stephan Puchegger (2005): Matrix and tensor notation in the
        theory of elasticity.

    .. [3] P. Helnwein (2001). Some Remarks on the Compressed Matrix
       Representation of Symmetric Second-Order and Fourth-Order Tensors.
       Computer Methods in Applied Mechanics and Engineering,
       190(22-23):2753-2770
    """
    if case == "stress":
        tensor = sym.Matrix([
            [vec[0], vec[5], vec[4]],
            [vec[5], vec[1], vec[3]],
            [vec[4], vec[3], vec[2]]])
    elif case == "strain":
        tensor = sym.Matrix([
            [vec[0], vec[5]/2, vec[4]/2],
            [vec[5]/2, vec[1], vec[3]/2],
            [vec[4]/2, vec[3]/2, vec[2]]])
    else:
        msg = "Case not supported.\n\n"\
            + "Available options are: 'stress' or 'strain'."
        raise ValueError(msg)
    return tensor


def rot_stress_voigt(Q):
    r"""Compute the rotation matrix for stresses in Voigt notation.

    In Voigt notation stresses transform as

    .. math::
        \sigma'_{I} = M_{IJ} \sigma_{J}

    where the coefficients define a 6x6 transformation matrix

    .. math::
        [M] =
        \begin{bmatrix}Q_{xx}^{2} & Q_{xy}^{2} & Q_{xz}^{2} &
        2 Q_{xy} Q_{xz} & 2 Q_{xx} Q_{xz} & 2 Q_{xx} Q_{xy}\\
        Q_{yx}^{2} & Q_{yy}^{2} & Q_{yz}^{2} & 2 Q_{yy} Q_{yz} &
        2 Q_{yx} Q_{yz} & 2 Q_{yx} Q_{yy}\\
        Q_{zx}^{2} & Q_{zy}^{2} & Q_{zz}^{2} & 2 Q_{zy} Q_{zz} &
        2 Q_{zx} Q_{zz} & 2 Q_{zx} Q_{zy}\\
        Q_{yx} Q_{zx} & Q_{yy} Q_{zy} & Q_{yz} Q_{zz} & Q_{yy}
        Q_{zz} + Q_{yz} Q_{zy} & Q_{yx} Q_{zz} + Q_{yz} Q_{zx}
        & Q_{yx} Q_{zy} + Q_{yy} Q_{zx}\\
        Q_{xx} Q_{zx} & Q_{xy} Q_{zy} & Q_{xz} Q_{zz} & Q_{xy}
        Q_{zz} + Q_{xz} Q_{zy} & Q_{xx} Q_{zz} + Q_{xz} Q_{zx}
        & Q_{xx} Q_{zy} + Q_{xy} Q_{zx}\\
        Q_{xx} Q_{yx} & Q_{xy} Q_{yy} & Q_{xz} Q_{yz} & Q_{xy}
        Q_{yz} + Q_{xz} Q_{yy} & Q_{xx} Q_{yz} + Q_{xz} Q_{yx}
        & Q_{xx} Q_{yy} + Q_{xy} Q_{yx}\end{bmatrix}

    and :math:`Q_{ij}` are the coefficients of the rotation matrix.

    Parameters
    ----------
    Q : Matrix (3,3)
        Rotation matrix.

    Returns
    -------
    M : Matrix (6,6)
        Rotation tensor in Voigt notation.

    References
    ----------
    .. [1] Auld, B. A. (1973). Acoustic fields and waves in solids
        (Vol. 1, p. 423). New York: Wiley. Sec 3.D.

    .. [2] Bower, Allan F. Applied mechanics of solids. CRC press, 2009.
        Ch. 3.

    Examples
    --------
    A rotation around the :math:`z`-axis of 30 degrees clockwise

    >>> theta = sym.pi/6
    >>> c = sym.cos(theta)
    >>> s = sym.sin(theta)
    >>> Q = sym.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    >>> M1 = rot_stress_voigt(Q)

    And the result should be

    .. math::
        \begin{bmatrix}
        \frac{3}{4} & \frac{1}{4} & 0 & 0 & 0 & \frac{\sqrt{3}}{2}\\
        \frac{1}{4} & \frac{3}{4} & 0 & 0 & 0 & - \frac{\sqrt{3}}{2}\\
        0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & \frac{\sqrt{3}}{2} & - \frac{1}{2} & 0\\
        0 & 0 & 0 & \frac{1}{2} & \frac{\sqrt{3}}{2} & 0\\
        - \frac{\sqrt{3}}{4} & \frac{\sqrt{3}}{4} & 0 & 0 & 0 & \frac{1}{2}
        \end{bmatrix}
    """
    M = sym.zeros(6)
    for i in range(3):
        for j in range(3):
            M[i, j] = Q[i, j]**2
            M[i, j + 3] = 2 * Q[i, (j + 1) % 3] * Q[i, (j + 2) % 3]
            M[i + 3, j] = Q[(i + 1) % 3, j] * Q[(i + 2) % 3, j]
            M[i + 3, j + 3] = Q[(i + 1) % 3, (j + 1) % 3] * \
                Q[(i + 2) % 3, (j + 2) % 3] + \
                Q[(i + 1) % 3, (j + 2) % 3] * \
                Q[(i + 2) % 3, (j + 1) % 3]
    return M


def rot_strain_voigt(Q):
    r"""Compute the Bond rotation matrix for strains in Voigt notation.

    In Voigt notation strains transform as

    .. math::
        \epsilon'_{K} = N_{KJ} \epsilon_{J}

    where the coefficients define a 6x6 transformation matrix

    .. math::
        [N] =
        \begin{bmatrix}Q_{xx}^{2} & Q_{xy}^{2} & Q_{xz}^{2} &
        Q_{xy} Q_{xz} & Q_{xx} Q_{xz} & Q_{xx} Q_{xy}\\
        Q_{yx}^{2} & Q_{yy}^{2} & Q_{yz}^{2} & Q_{yy} Q_{yz} &
        Q_{yx} Q_{yz} & Q_{yx} Q_{yy}\\
        Q_{zx}^{2} & Q_{zy}^{2} & Q_{zz}^{2} & Q_{zy} Q_{zz} &
        Q_{zx} Q_{zz} & Q_{zx} Q_{zy}\\
        2 Q_{yx} Q_{zx} & 2 Q_{yy} Q_{zy} & 2 Q_{yz} Q_{zz} &
        Q_{yy} Q_{zz} + Q_{yz} Q_{zy} & Q_{yx} Q_{zz} +
        Q_{yz} Q_{zx} & Q_{yx} Q_{zy} + Q_{yy} Q_{zx}\\
        2 Q_{xx} Q_{zx} & 2 Q_{xy} Q_{zy} & 2 Q_{xz} Q_{zz} &
        Q_{xy} Q_{zz} + Q_{xz} Q_{zy} & Q_{xx} Q_{zz} +
        Q_{xz} Q_{zx} & Q_{xx} Q_{zy} + Q_{xy} Q_{zx}\\
        2 Q_{xx} Q_{yx} & 2 Q_{xy} Q_{yy} & 2 Q_{xz} Q_{yz} &
        Q_{xy} Q_{yz} + Q_{xz} Q_{yy} & Q_{xx} Q_{yz} +
        Q_{xz} Q_{yx} & Q_{xx} Q_{yy} + Q_{xy} Q_{yx}\end{bmatrix}

    and :math:`Q_{ij}` are the coefficients of the rotation matrix.

    Parameters
    ----------
    Q : Matrix (3,3)
        Rotation matrix

    Returns
    -------
    N : Matrix (6,6)
        Rotation tensor in Voigt notation.

    References
    ----------
    .. [1] Auld, B. A. (1973). Acoustic fields and waves in solids
        (Vol. 1, p. 423). New York: Wiley. Sec 3.D.

    .. [2] Bower, Allan F. Applied mechanics of solids. CRC press, 2009.
        Ch. 3.

    Examples
    --------
    A rotation around the :math:`z`-axis of 30 degrees clockwise

    >>> theta = sym.pi/6
    >>> c = sym.cos(theta)
    >>> s = sym.sin(theta)
    >>> Q = sym.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    >>> M1 = rot_strain_voigt(Q)

    And the result should be

    .. math::
        \begin{bmatrix}
        \frac{3}{4} & \frac{1}{4} & 0 & 0 & 0 & \frac{\sqrt{3}}{4}\\
        \frac{1}{4} & \frac{3}{4} & 0 & 0 & 0 & - \frac{\sqrt{3}}{4}\\
        0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & \frac{\sqrt{3}}{2} & - \frac{1}{2} & 0\\
        0 & 0 & 0 & \frac{1}{2} & \frac{\sqrt{3}}{2} & 0\\
        -\frac{\sqrt{3}}{2} & \frac{\sqrt{3}}{2} & 0 & 0 & 0 & \frac{1}{2}
        \end{bmatrix}

    """
    N = sym.zeros(6)
    for i in range(3):
        for j in range(3):
            N[i, j] = Q[i, j]**2
            N[i, j + 3] = Q[i, (j + 1) % 3] * Q[i, (j + 2) % 3]
            N[i + 3, j] = 2 * Q[(i + 1) % 3, j] * Q[(i + 2) % 3, j]
            N[i + 3, j + 3] = Q[(i + 1) % 3, (j + 1) % 3] * \
                Q[(i + 2) % 3, (j + 2) % 3] + \
                Q[(i + 1) % 3, (j + 2) % 3] * \
                Q[(i + 2) % 3, (j + 1) % 3]
    return N


def rot_stiff_voigt(C, Q):
    r"""
    Compute Bond rotation matrix for stiffness tensor in Voigt
    notation.

    In Voigt notation stiffness tensors transform as

    .. math::
        [C'] = [M][C][M]^T

    or

    .. math::
        C'_{HK} = M_{HI}M_{KJ}C_{IJ}

    where the coefficients define a 6x6 transformation matrix

    .. math::
        [M] =
        \begin{bmatrix}Q_{xx}^{2} & Q_{xy}^{2} & Q_{xz}^{2} &
        2 Q_{xy} Q_{xz} & 2 Q_{xx} Q_{xz} & 2 Q_{xx} Q_{xy}\\
        Q_{yx}^{2} & Q_{yy}^{2} & Q_{yz}^{2} & 2 Q_{yy} Q_{yz} &
        2 Q_{yx} Q_{yz} & 2 Q_{yx} Q_{yy}\\
        Q_{zx}^{2} & Q_{zy}^{2} & Q_{zz}^{2} & 2 Q_{zy} Q_{zz} &
        2 Q_{zx} Q_{zz} & 2 Q_{zx} Q_{zy}\\
        Q_{yx} Q_{zx} & Q_{yy} Q_{zy} & Q_{yz} Q_{zz} & Q_{yy}
        Q_{zz} + Q_{yz} Q_{zy} & Q_{yx} Q_{zz} + Q_{yz} Q_{zx}
        & Q_{yx} Q_{zy} + Q_{yy} Q_{zx}\\
        Q_{xx} Q_{zx} & Q_{xy} Q_{zy} & Q_{xz} Q_{zz} & Q_{xy}
        Q_{zz} + Q_{xz} Q_{zy} & Q_{xx} Q_{zz} + Q_{xz} Q_{zx}
        & Q_{xx} Q_{zy} + Q_{xy} Q_{zx}\\
        Q_{xx} Q_{yx} & Q_{xy} Q_{yy} & Q_{xz} Q_{yz} & Q_{xy}
        Q_{yz} + Q_{xz} Q_{yy} & Q_{xx} Q_{yz} + Q_{xz} Q_{yx}
        & Q_{xx} Q_{yy} + Q_{xy} Q_{yx}\end{bmatrix}

    and :math:`Q_{ij}` are the coefficients of the rotation matrix.

    Parameters
    ----------
    C : Matrix (6,6)
        Stiffness tensor in Voigt notation.
    Q : Matrix (3,3)
        Rotation matrix.

    Returns
    -------
    Cp : Matrix (6,6)
        Rotated stiffness tensor in Voigt notation.

    References
    ----------
    .. [1] Auld, B. A. (1973). Acoustic fields and waves in solids
        (Vol. 1, p. 423). New York: Wiley. Sec 3.D.

    .. [2] Bower, Allan F. Applied mechanics of solids. CRC press, 2009.
        Ch. 3.

    Examples
    --------
    Let us start with a rotation around the :math:`z`-axis of 30 degrees
    clockwise for an isotropic material, the rotated matrix should be
    the same as the original.

    >>> theta = sym.pi/6
    >>> c = sym.cos(theta)
    >>> s = sym.sin(theta)
    >>> Q = sym.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    >>> C = iso_stiff(1, sym.S(1)/3)
    >>> Cp = rot_stiff_voigt(C, Q)
    >>> C.equals(Cp)
    True
    """
    M = rot_stress_voigt(Q)
    Cp = M * C * M.T
    return Cp


def rot_compl_voigt(S, Q):
    r"""Compute rotation matrix for compliance tensor in Voigt notation.

    In Voigt notation compliance tensors transform as

    .. math::
        [S'] = [N][S][N]^T

    or

    .. math::
        S'_{HK} = N_{HI}N_{KJ}S_{IJ}

    where the coefficients define a 6x6 transformation matrix

    .. math::
        [N] =
        \begin{bmatrix}Q_{xx}^{2} & Q_{xy}^{2} & Q_{xz}^{2} &
        Q_{xy} Q_{xz} & Q_{xx} Q_{xz} & Q_{xx} Q_{xy}\\
        Q_{yx}^{2} & Q_{yy}^{2} & Q_{yz}^{2} & Q_{yy} Q_{yz} &
        Q_{yx} Q_{yz} & Q_{yx} Q_{yy}\\
        Q_{zx}^{2} & Q_{zy}^{2} & Q_{zz}^{2} & Q_{zy} Q_{zz} &
        Q_{zx} Q_{zz} & Q_{zx} Q_{zy}\\
        2 Q_{yx} Q_{zx} & 2 Q_{yy} Q_{zy} & 2 Q_{yz} Q_{zz} &
        Q_{yy} Q_{zz} + Q_{yz} Q_{zy} & Q_{yx} Q_{zz} +
        Q_{yz} Q_{zx} & Q_{yx} Q_{zy} + Q_{yy} Q_{zx}\\
        2 Q_{xx} Q_{zx} & 2 Q_{xy} Q_{zy} & 2 Q_{xz} Q_{zz} &
        Q_{xy} Q_{zz} + Q_{xz} Q_{zy} & Q_{xx} Q_{zz} +
        Q_{xz} Q_{zx} & Q_{xx} Q_{zy} + Q_{xy} Q_{zx}\\
        2 Q_{xx} Q_{yx} & 2 Q_{xy} Q_{yy} & 2 Q_{xz} Q_{yz} &
        Q_{xy} Q_{yz} + Q_{xz} Q_{yy} & Q_{xx} Q_{yz} +
        Q_{xz} Q_{yx} & Q_{xx} Q_{yy} + Q_{xy} Q_{yx}\end{bmatrix}

    and :math:`Q_{ij}` are the coefficients of the rotation matrix.

    Parameters
    ----------
    S : Matrix (6,6)
        Compliance tensor in Voigt notation.
    Q : Matrix (3,3)
        Rotation matrix.

    Returns
    -------
    Cp : Matrix (6,6)
        Rotated compliance tensor in Voigt notation.

    References
    ----------
    .. [1] Auld, B. A. (1973). Acoustic fields and waves in solids
        (Vol. 1, p. 423). New York: Wiley. Sec 3.D.

    .. [2] Bower, Allan F. Applied mechanics of solids. CRC press, 2009.
        Ch. 3.

    Examples
    --------
    Let us take a rotation around the :math:`z`-axis of 30 degrees
    clockwise for an isotropic material, the rotated matrix should be
    the same as the original.

    >>> theta = sym.pi/6
    >>> c = sym.cos(theta)
    >>> s = sym.sin(theta)
    >>> Q = sym.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    >>> S = iso_compl(1, sym.S(1)/3)
    >>> Sp = rot_compl_voigt(S, Q)
    >>> S.equals(Sp)
    True

    """
    N = rot_strain_voigt(Q)
    Sp = N * S * N.T
    return Sp


#%% Kelvin-Mandel notation
def mat2mandel(tensor):
    """Convert from matrix notation to Kelvin-Mandel notation.

    Parameters
    ----------
    tensor : Matrix (3, 3)
        Second rank tensor in matrix notation.

    Returns
    -------
    vec : Matrix (6, 1)
        Vector-like representation of the tensor.

    References
    ----------
    .. [1] Voigt notation. In Wikipedia, The Free Encyclopedia.
        Retrieved from
        http://en.wikipedia.org/w/index.php?title=Voigt_notation

    .. [2] Stephan Puchegger (2005): Matrix and tensor notation in the
        theory of elasticity.

    .. [3] P. Helnwein (2001). Some Remarks on the Compressed Matrix
       Representation of Symmetric Second-Order and Fourth-Order Tensors.
       Computer Methods in Applied Mechanics and Engineering,
       190(22-23):2753-2770
    """
    vec = sym.Matrix([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                     sym.sqrt(2)*tensor[1, 2],
                     sym.sqrt(2)*tensor[0, 2],
                     sym.sqrt(2)*tensor[0, 1]])
    return vec


def mandel2mat(vec):
    """Convert from Mandel's notation to matrix.

    Parameters
    ----------
    vec : Matrix (6, 1)
        Vector-like representation of the tensor.

    Returns
    -------
    tensor : Matrix (3, 3)
        Second rank tensor in matrix notation.

    References
    ----------
    .. [1] Voigt notation. In Wikipedia, The Free Encyclopedia.
        Retrieved from
        http://en.wikipedia.org/w/index.php?title=Voigt_notation

    .. [2] Stephan Puchegger (2005): Matrix and tensor notation in the
        theory of elasticity.

    .. [3] P. Helnwein (2001). Some Remarks on the Compressed Matrix
       Representation of Symmetric Second-Order and Fourth-Order Tensors.
       Computer Methods in Applied Mechanics and Engineering,
       190(22-23):2753-2770
    """
    tensor = sym.Matrix([
        [vec[0], vec[5]/sym.sqrt(2), vec[4]/sym.sqrt(2)],
        [vec[5]/sym.sqrt(2), vec[1], vec[3]/sym.sqrt(2)],
        [vec[4]/sym.sqrt(2), vec[3]/sym.sqrt(2), vec[2]]])
    return tensor


def rot_mandel(Q):
    r"""Compute the rotation matrix in Kelvin-Mandel notation.

    In Mandel notation stresses/strains transform as

    .. math::
        \sigma'_{I} = M_{IJ} \sigma_{J}

    where the coefficients define a 6x6 transformation matrix

    .. math::
        [M] =
        \begin{bmatrix}
        Q_{xx}^{2} & Q_{xy}^{2} & Q_{xz}^{2} &
        \sqrt{2} Q_{xy} Q_{xz} & \sqrt{2} Q_{xx} Q_{xz} & \sqrt{2} Q_{xx} Q_{xy}\\
        Q_{yx}^{2} & Q_{yy}^{2} & Q_{yz}^{2} & \sqrt{2} Q_{yy} Q_{yz} &
        \sqrt{2} Q_{yx} Q_{yz} & \sqrt{2} Q_{yx} Q_{yy}\\
        Q_{zx}^{2} & Q_{zy}^{2} & Q_{zz}^{2} & \sqrt{2} Q_{zy} Q_{zz} &
        \sqrt{2} Q_{zx} Q_{zz} & \sqrt{2} Q_{zx} Q_{zy}\\
        \sqrt{2}Q_{yx} Q_{zx} & \sqrt{2}Q_{yy} Q_{zy} & \sqrt{2}Q_{yz} Q_{zz}
        & Q_{yy}Q_{zz} + Q_{yz} Q_{zy} & Q_{yx} Q_{zz} + Q_{yz} Q_{zx}
        & Q_{yx} Q_{zy} + Q_{yy} Q_{zx}\\
        \sqrt{2}Q_{xx} Q_{zx} & \sqrt{2}Q_{xy} Q_{zy} & \sqrt{2}Q_{xz} Q_{zz}
        & Q_{xy}Q_{zz} + Q_{xz} Q_{zy} & Q_{xx} Q_{zz} + Q_{xz} Q_{zx}
        & Q_{xx} Q_{zy} + Q_{xy} Q_{zx}\\
        \sqrt{2}Q_{xx} Q_{yx} & \sqrt{2}Q_{xy} Q_{yy} & \sqrt{2}Q_{xz} Q_{yz}
        & Q_{xy} Q_{yz} + Q_{xz} Q_{yy} & Q_{xx} Q_{yz} + Q_{xz} Q_{yx}
        & Q_{xx} Q_{yy} + Q_{xy} Q_{yx}
        \end{bmatrix}

    and :math:`Q_{ij}` are the coefficients of the rotation matrix.

    Parameters
    ----------
    Q : Matrix (3,3)
        Rotation matrix.

    Returns
    -------
    M : Matrix (6,6)
        Rotation tensor in Kelvin-Mandel notation.

    References
    ----------
    .. [1] Mehrabadi, M. M., & Cowin, S. C. (1990). Eigentensors of
       linear anisotropic elastic materials. The Quarterly Journal of
       Mechanics and Applied Mathematics, 43(1), 15-41.

    Examples
    --------
    A rotation around the :math:`z`-axis of 30 degrees clockwise

    >>> theta = sym.pi/6
    >>> c = sym.cos(theta)
    >>> s = sym.sin(theta)
    >>> Q = sym.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    >>> M1 = rot_mandel(Q)

    And the result should be

    .. math::
        \begin{bmatrix}
        \frac{3}{4} & \frac{1}{4} & 0 & 0 & 0 & \frac{\sqrt{6}}{4}\\
        \frac{1}{4} & \frac{3}{4} & 0 & 0 & 0 & - \frac{\sqrt{6}}{4}\\
        0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & \frac{\sqrt{3}}{2} & - \frac{1}{2} & 0\\
        0 & 0 & 0 & \frac{1}{2} & \frac{\sqrt{3}}{2} & 0\\
        - \frac{\sqrt{6}}{4} & \frac{\sqrt{6}}{4} & 0 & 0 & 0 & \frac{1}{2}
        \end{bmatrix}
    """
    M = sym.zeros(6)
    for i in range(3):
        for j in range(3):
            M[i, j] = Q[i, j]**2
            M[i, j + 3] = sym.sqrt(2) * Q[i, (j + 1) % 3] * Q[i, (j + 2) % 3]
            M[i + 3, j] = sym.sqrt(2) * Q[(i + 1) % 3, j] * Q[(i + 2) % 3, j]
            M[i + 3, j + 3] = Q[(i + 1) % 3, (j + 1) % 3] * \
                Q[(i + 2) % 3, (j + 2) % 3] + \
                Q[(i + 1) % 3, (j + 2) % 3] * \
                Q[(i + 2) % 3, (j + 1) % 3]
    return M


#%% Fourth order
def christ_stiff(C, n, numeric=False):
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
    if numeric:
        Gamma = np.zeros((3, 3))
    else:
        Gamma = sym.zeros(3)
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()