# -*- coding: utf-8 -*-
"""
Visualization module
--------------------

Functions to aid the visualization of mathematical
entities such as second rank tensors.

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import eigvalsh
from continuum_mechanics.tensor import christ_stiff

# Plotting configuration
gray = '#757575'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["text.color"] = gray
fontsize = plt.rcParams["font.size"] = 12
plt.rcParams["xtick.color"] = gray
plt.rcParams["ytick.color"] = gray
plt.rcParams["axes.labelcolor"] = gray
plt.rcParams["axes.edgecolor"] = gray
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


#%% Mohr circles
def mohr2d(stress, ax=None):
    """Plot Mohr circle for a 2D tensor
    
    Parameters
    ----------
    stress : ndarray
        Stress tensor.
    ax : Matplotlib axes, optional
        Axes where the plot is going to be added.

    References
    ----------
    .. [BRAN] Brannon, R. (2003). Mohr’s Circle and more circles.
        Poslední revize, 29(10).
    
    """
    try:
        stress = np.asarray(stress).astype(float)
        stress.shape = 2, 2
    except:
        TypeError("Stress should be represented as an array.")
    skew = (stress - stress.T)/2
    sym = (stress + stress.T)/2
    S11, S12, S21, S22 = stress.flatten()
    mean = sym.trace()/2
    center = [mean, skew[1, 0]]
    radius = np.sqrt((sym[0, 0] - sym[1, 1])**2/4 + sym[0, 1]**2)
    Smin = center[0] - radius
    Smax = center[0] + radius
    
    if ax is None:
        plt.figure()
        ax = plt.gca() 
    circ = plt.Circle(center, radius, facecolor='#cce885', lw=3,
    edgecolor='#5c8037') 
    plt.axis('image')    
    ax.add_artist(circ)
    ax.set_xlim(Smin - .1*radius, Smax + .1*radius)
    ax.set_ylim(center[1] - 1.1*radius, center[1] + 1.1*radius)
    plt.plot([S22, S11], [S21, -S12], 'ko')
    plt.plot([S22, S11], [S21, -S12], 'k')
    plt.plot(center[0], center[1], 'o', mfc='w')
    plt.text(S22 + 0.1*radius, S21, 'A')
    plt.text(S11 + 0.1*radius, -S12, 'B')
    plt.xlabel(r"$\sigma$", size=fontsize + 2)
    plt.ylabel(r"$\tau$", size=fontsize + 2)
    return ax


def mohr3d(stress, ax=None):
    r"""Plot 3D Mohr circles

    Parameters
    ----------
    stress : ndarray
        Stress tensor.
    ax : Matplotlib axes, optional
        Axes where the plot is going to be added.

    """
    try:
        stress = np.asarray(stress).astype(float)
        stress.shape = 3, 3
    except:
        TypeError("Stress should be represented as an array.")        
    S3, S2, S1 = eigvalsh(stress)

    R_maj = 0.5*(S1 - S3)
    cent_maj = 0.5*(S1+S3)
    
    R_min = 0.5*(S2 - S3)
    cent_min = 0.5*(S2 + S3)
    
    R_mid = 0.5*(S1 - S2)
    cent_mid = 0.5*(S1 + S2)
    
    if ax is None:
        plt.figure()
        ax = plt.gca() 
    circ1 = plt.Circle((cent_maj,0), R_maj, facecolor='#cce885', lw=3,
                       edgecolor='#5c8037')
    circ2 = plt.Circle((cent_min,0), R_min, facecolor='w', lw=3,
                       edgecolor='#15a1bd')
    circ3 = plt.Circle((cent_mid,0), R_mid, facecolor='w', lw=3,
                       edgecolor='#e4612d')
    plt.axis('image')
    ax.add_artist(circ1)
    ax.add_artist(circ2)
    ax.add_artist(circ3)
    ax.set_xlim(S3 - .1*R_maj, S1 + .1*R_maj)
    ax.set_ylim(-1.1*R_maj, 1.1*R_maj)
    plt.xlabel(r"$\sigma$", size=fontsize + 2)
    plt.ylabel(r"$\tau$", size=fontsize + 2)
    return ax


#%% Tensor visualizations
def traction_circle(stress, npts=48, ax=None):
    """
    Visualize a second order tensor as a collection of
    tractions vectors over a circle.

    Parameters
    ----------
    stress : ndarray
        Stress tensor.
    npts : int, optional
        Number of vector to plot over the circle.
    ax : Matplotlib axes, optional
        Axes where the plot is going to be added.    
    """
    try:
        stress = np.asarray(stress).astype(float)
        stress.shape = 2, 2
    except:
        TypeError("Stress should be represented as an array.")
    rad = 1
    theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
    nx = np.cos(theta)
    ny = np.sin(theta)
    vec = np.vstack((nx, ny))
    tracciones = stress.dot(vec)
    if ax is None:
        plt.figure()
        ax = plt.gca()
    plt.plot(rad * nx, rad * ny, alpha=0.5, color="black", zorder=4)
    plt.quiver(rad * nx, rad * ny, 
               nx, ny, alpha=0.3, scale=10, zorder=3)
    plt.quiver(rad * nx, rad * ny, 
               tracciones[0, :], tracciones[1, :],
               np.sqrt (tracciones[0, :]**2 + tracciones[1, :]**2),
               scale=30, cmap="Reds", zorder=5)
    plt.axis("image")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r"$x$", size=fontsize + 2)
    plt.ylabel(r"$y$", size=fontsize + 2)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    return ax


## Fourth order vis
def christofel_eig(C, nphi=30, nth=30):
    r"""Compute surfaces of eigenvalues for the Christoffel stiffness

    For every direction

    .. math::
      \mathbf{n} =(\sin(\theta)\cos(\phi),\sin(\theta)\sin(\phi),\cos(\theta))

    computes the eigenvalues of the Christoffel stiffness tensor. These
    eigencalues are :math:`\rho v_p^2`, where :math:`\rho` is the mass
    density and :math:`v_p` is the phase speed.

    Parameters
    ----------
    C : (6,6) array
        Stiffness tensor in Voigt notation.
    nphi : (optional) int
        Number of partitions in the azimut angle (phi).
    nth : (optional) int
        Number of partitions in the cenit angle (theta).

    Returns
    -------
    V1, V2, V3 : (nphi, nth) arrays
        Eigenvalues for the desired discretization.
    phi_vec : (nphi) array
        Array with azimut angles.
    theta_vec : (nth) array
        Array with cenit angles.

    """
    phi_vec, theta_vec = np.mgrid[0: 2*np.pi: nphi*1j, 0: np.pi: nth*1j]
    V1 = 0*phi_vec
    V2 = 0*phi_vec
    V3 = 0*phi_vec

    for i in range(nphi):
        for j in range(nth):
            phi = phi_vec[i, j]
            theta = theta_vec[i, j]
            n = [np.sin(theta)*np.cos(phi),
                 np.sin(theta)*np.sin(phi),
                 np.cos(theta)]
            Gamma = christ_stiff(C, n)
            vals = eigvalsh(Gamma)
            V1[i, j] = vals[0]
            V2[i, j] = vals[1]
            V3[i, j] = vals[2]

    return V1, V2, V3, phi_vec, theta_vec


def plot_surf(R, phi, theta, title="Wave speed", **kwargs):
    r"""Plot the surface given by R(phi, theta).

    Parameters
    ----------
    R : (m,n) ndarray
        Radius function.
    phi_vec : (m,n) ndarray
        Meshgrid for the azimut angle (phi).
    theta_vec : (m,n) ndarray
        Meshgrid for the cenit angle (theta).
    **kwargs : keyword arguments (optional)
        Keyword arguments for `mlab.mesh`.

    Returns
    -------
    surf : mayavi mesh
        Mayavi mesh for the surface `R(phi, theta)`.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = R * np.cos(phi) * np.sin(theta)
    Y = R * np.sin(phi) * np.sin(theta)
    Z = R * np.cos(theta)
    FC = np.sqrt(X * X + Y * Y + Z * Z)

    # Set colormap bounds
    vmax = FC.max()
    vmin = FC.min()
    FC = (FC - vmin) / (vmax - vmin)

    # Fix aspect ratio
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(),
                          Z.max() - Z.min()]).max() / 2.0
    mean_x = X.mean()
    mean_y = Y.mean()
    mean_z = Z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)

    ax.plot_surface(X, Y, Z, facecolors=plt.cm.magma(FC),
                    rstride=1, cstride=1, linewidth=0,
                    antialiased=False)
    return ax


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # beta-brass
    C11 = 52
    C12 = 27.5
    C44 = 173
    rho = 7600
    C = np.zeros((6, 6))
    C[0:3, 0:3] = np.array([[C11, C12, C12],
                            [C12, C11, C12],
                            [C12, C12, C11]])
    C[3:6, 3:6] = np.diag([C44, C44, C44])
    
    
    # Phi is the azimut angle
    # theta is the cenital angle
    V1, V2, V3, phi_vec, theta_vec = christofel_eig(C, 100, 100) 
    V1 = np.sqrt(V1*1e9/rho)
    V2 = np.sqrt(V2*1e9/rho)
    V3 = np.sqrt(V3*1e9/rho)

    plot_surf(V1, phi_vec, theta_vec)
    plot_surf(V2, phi_vec, theta_vec)
    plot_surf(V3, phi_vec, theta_vec)
    plt.show()
