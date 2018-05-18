# -*- coding: utf-8 -*-
"""
Visualization module
--------------------

Functions to aid the visualization of mathematical
entities such as second rank tensors.

"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

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
    
    """
    try:
        stress = np.asarray(stress).astype(float)
        stress.shape = 2, 2
    except:
        TypeError("Stress should be represented as an array.")
    S11 = stress[0, 0]
    S12 = stress[0, 1]
    S22 = stress[1, 1]
    center = [(S11 + S22)/2.0, 0.0]
    radius = np.sqrt((S11 - S22)**2/4.0 + S12**2)
    Smin = center[0] - radius
    Smax = center[0] + radius
    
    if ax is None:
        plt.figure()
        ax = plt.gca() 
    circ = plt.Circle((center[0],0), radius, facecolor='#cce885', lw=3,
    edgecolor='#5c8037') 
    plt.axis('image')    
    ax.add_artist(circ)
    ax.set_xlim(Smin - .1*radius, Smax + .1*radius)
    ax.set_ylim(-1.1*radius, 1.1*radius)
    plt.plot([S22, S11], [S12, -S12], 'ko')
    plt.plot([S22, S11], [S12, -S12], 'k')
    plt.plot(center[0], center[1], 'o', mfc='w')
    plt.text(S22 + 0.1*radius, S12, 'A')
    plt.text(S11 + 0.1*radius, -S12, 'B')
    plt.xlabel(r"$\sigma$", size=fontsize + 2)
    plt.ylabel(r"$\tau$", size=fontsize + 2)


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



if __name__ == "__main__":
    import doctest
    doctest.testmod()