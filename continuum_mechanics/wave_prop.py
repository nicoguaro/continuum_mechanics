# -*- coding: utf-8 -*-
"""
Wave propagation module
------------------------

Wave propagation in elastic media
"""
from __future__ import division, print_function
from numpy import cos, sin, exp, pi, arcsin
from numpy import array


def scatter_matrix(alpha, beta, ang_i, ang_j):
    r"""
    Scatter matrix for reflection/conversion coefficients
    in an elastic halfspace
    
    The matrix is written as presented in [AKI2009]_.
    
    .. math::

        \begin{bmatrix}
            \acute{P}\grave{P} &\acute{P}\grave{S}\\
            \acute{S}\grave{P} &\acute{S}\grave{S}
        \end{bmatrix}

    Parameters
    ----------
    alpha : float
        Speed for the P-wave.
    beta : float
        Speed for the S-wave.
    ang_i : ndarray
        Incidence angle for the P-wave.
    ang_j : ndarray
        Incidence angle for the S-wave.

    Returns
    -------
    scatter : ndarray
        Scatter matrix for the reflection conversion modes.

    References
    ----------
    
    .. [AKI2009] Keiiti Aki and Paul G. Richards. Quantitative
        Seismology. University Science Books, 2009.
    """
    p = sin(ang_i)/alpha
    p1 = cos(ang_i)/alpha
    p2 = cos(ang_j)/beta
    denom = (1/beta**2 - 2*p**2)**2 + 4*p**2*p1*p2
    PP = -(1/beta**2 - 2*p**2)**2 + 4*p**2*p1*p2
    PS = 4*alpha/beta*p*p1*(1/beta**2 - 2*p**2)
    SP = 4*beta/alpha*p*p2*(1/beta**2 - 2*p**2)
    SS = (1/beta**2 - 2*p**2)**2 - 4*p**2*p1*p2
    scatter = array([
                [PP/denom, PS/denom],
                [SP/denom, SS/denom]])
    return scatter


def scatter_matrix_micropolar(alpha, beta, ang_i, ang_j):
    r"""
    Scatter matrix for reflection/conversion coefficients
    in an elastic halfspace
    
    The matrix is written as presented in [AKI2009]_.
    
    .. math::

        \begin{bmatrix}
            \acute{P}\grave{P} &\acute{P}\grave{S}\\
            \acute{S}\grave{P} &\acute{S}\grave{S}
        \end{bmatrix}

    Parameters
    ----------
    alpha : float
        Speed for the P-wave.
    beta : float
        Speed for the S-wave.
    ang_i : ndarray
        Incidence angle for the P-wave.
    ang_j : ndarray
        Incidence angle for the S-wave.

    Returns
    -------
    scatter : ndarray
        Scatter matrix for the reflection conversion modes.

    References
    ----------
    
    .. [AKI2009] Keiiti Aki and Paul G. Richards. Quantitative
        Seismology. University Science Books, 2009.
    """
    p = sin(ang_i)/alpha
    p1 = cos(ang_i)/alpha
    p2 = cos(ang_j)/beta
    denom = (1/beta**2 - 2*p**2)**2 + 4*p**2*p1*p2
    PP = -(1/beta**2 - 2*p**2)**2 + 4*p**2*p1*p2
    PS = 4*alpha/beta*p*p1*(1/beta**2 - 2*p**2)
    SP = 4*beta/alpha*p*p2*(1/beta**2 - 2*p**2)
    SS = (1/beta**2 - 2*p**2)**2 - 4*p**2*p1*p2
    scatter = array([
                [PP/denom, PS/denom],
                [SP/denom, SS/denom]])
    return scatter

def ricker_spectrum(freq, freq_c):
    r"""Spectrum of the Ricker wavelet
    
    The spectrum is given by
    
    .. math::
        
        \hat{f}(\xi) = -\xi^2 e^{-\xi^2}

    where :math:`\xi= f/f_c` is the dimensionless frequency, and 
    :math:`f_c` is the central frequency. The time between peaks
    is given by
    
    .. math::

        t_\text{peaks} = \frac{\sqrt{6}}{\pi f_c}\, .
 
    For further reference see [KIM1991]_ and [RIC1945]_.
    
    Parameters
    ----------
    freq : ndarray
        Frequency.
    freq_c : float
        Frequency for the peak.

    Returns
    -------
    amplitude : ndarray
        Fourier amplitude for the given frequencies.

    References
    ----------
    
    .. [KIM1991] A. Papageorgiou and J. Kim. Study of the propagation
        of seismic waves in Caracas Valley with reference to the
        29 July 1967 earthquake: SH waves. Bulletin of the
        Seismological Society of America, 1991, 81 (6): 2214-223.
    .. [RIC1945] N. Ricker. The computation of output disturbances from
        from amplifiers for true wavelets inputs. Geophysics,
        10: 207-220.
    """
    xi = freq/freq_c
    return -xi**2*exp(-xi**2)


def ricker_signal(t, freq_c):
    r"""Ricker wavelet in the time domain
    
    The signal is given by
    
    .. math::
        
        f(\tau) = (2\tau^2 - 1) e^{-\tau^2}

    where :math:`\tau=\pi f_c t` is the dimensionless time, and
    :math:`f_c` is the central frequency. The time between peaks
    is given by
    
    .. math::

        t_\text{peaks} = \frac{\sqrt{6}}{\pi f_c}\, .
 
    For further reference see [KIM1991]_.
    
    Parameters
    ----------
    t : ndarray
        Time.
    freq_c : float
        Central frequency for the signal.

    Returns
    -------
    signal : ndarray
        Signal evaluated at given times.

    References
    ----------
    
    .. [KIM1991] A. Papageorgiou and J. Kim. Study of the propagation
        of seismic waves in Caracas Valley with reference to the
        29 July 1967 earthquake: SH waves. Bulletin of the
        Seismological Society of America, 1991, 81 (6): 2214-223.
    .. [RIC1945] N. Ricker. The computation of output disturbances from
        from amplifiers for true wavelets inputs. Geophysics,
        10: 207-220.
    """
    tau = pi*t*freq_c
    return (2*tau**2 - 1)*exp(-tau**2)


def disp_incident_P(amp, omega, alpha, beta, ang_i, x, z):
    """Displacement for a P-wave incidence

     Parameters
    ----------
    amp : ndarray
        Amplitude for given frequencies.
    omega : ndarray
        Angular frequency.
    alpha : float
        Speed of the P wave.
    beta : float
        Speed of the S wave.
    ang_i : float
        Incidence angle.
    x : ndarray
        Horizontal coordinate.
    z : ndarray
        Vertical coordinate.

    Returns
    -------
    u : ndarray
        Spectrum for the horizontal component of displacement.
    v : ndarray
        Spectrum for the vertical component of displacement.
    
    """
    p = sin(ang_i)/alpha
    ang_j = arcsin(beta * p)
    p1 = cos(ang_i)/alpha
    p2 = cos(ang_j)/beta
    scatter = scatter_matrix(alpha, beta, ang_i, ang_j) 
    PP = scatter[0, 0]
    PS = scatter[0, 1]
    
    # Horizontal
    u_P_in = amp * sin(ang_i) * exp(1j*omega*(p*x - p1*z))
    u_P_ref = amp * sin(ang_i) * PP * exp(1j*omega*(p*x + p1*z))
    u_S_ref = amp * cos(ang_j) * PS * exp(1j*omega*(p*x + p2*z))
    
    # Vertical
    v_P_in = -amp * cos(ang_i) * exp(1j*omega*(p*x - p1*z))
    v_P_ref = amp * cos(ang_i) * PP * exp(1j*omega*(p*x + p1*z))
    v_S_ref = -amp * sin(ang_j) * PS * exp(1j*omega*(p*x + p2*z))
    
    # Total
    u = u_P_in + u_P_ref + u_S_ref
    v = v_P_in + v_P_ref + v_S_ref
    return u, v
    
    


if __name__ ==  "__main__":
    import doctest
    doctest.testmod()