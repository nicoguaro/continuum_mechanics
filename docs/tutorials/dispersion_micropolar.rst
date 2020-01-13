===========================================
Dispersion relations in a micropolar medium
===========================================

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/nicoguaro/continuum_mechanics/master?filepath=docs%2Ftutorials%2Fdispersion_micropolar.ipynb

We are interested in computing the dispersion relations in a homogeneous
micropolar solid.

Wave propagation in micropolar solids
-------------------------------------

The equations of motion for a micropolar solid are given by [NOWACKI]_,
[GUARIN]_

.. math::

  \begin{align}
  &c_1^2
  \nabla\nabla\cdot\mathbf{u}- c_2^2\nabla\times\nabla\times\mathbf{u} + K^2\nabla\times\boldsymbol{\theta} = -\omega^2 \mathbf{u} \, ,\\
  &c_3^2 \nabla\nabla\cdot\boldsymbol{\theta} - c_4^2\nabla\times\nabla\times\boldsymbol{\theta} + Q^2\nabla\times\mathbf{u} - 2Q^2\boldsymbol{\theta} = -\omega^2 \boldsymbol{\theta} \,
  \end{align}

where :math:`\mathbf{u}` is the displacement vector and
:math:`\boldsymbol{\theta}` is the microrrotations vector, and where:
:math:`c_1` represents the phase/group speed for the longitudinal wave
(:math:`P`) that is non-dispersive as in the classical case, :math:`c_2`
represents the high-frequency limit phase/group speed for a transverse
wave (:math:`S`) that is dispersive unlike the classical counterpart,
:math:`c_3` represents the high-frequency limit phase/group speed for a
longitudinal-rotational wave (:math:`LR`) with a corkscrew-like motion
that is dispersive and does not have a classical counterpart,
:math:`c_4` represents the high-frequency limit phase/group speed for a
transverse-rotational wave (:math:`TR`) that is dispersive and does not
have a classical counterpart, :math:`Q` represents the cut-off frequency
for rotational waves appearance, and :math:`K` quantifies the difference
between the low-frequency and high-frequency phase/group speed for the
S-wave. These parameters are defined by:

.. math::

  \begin{align}
  c_1^2 = \frac{\lambda +2\mu}{\rho},\quad &c_3^2 =\frac{\beta + 2\eta}{J},\\
  c_2^2 = \frac{\mu +\alpha}{\rho},\quad &c_4^2 =\frac{\eta + \varepsilon}{J},\\
  Q^2= \frac{2\alpha}{J},\quad &K^2 =\frac{2\alpha}{\rho} \, ,
  \end{align}

Dispersion relations
--------------------

To identify types of propagating waves that can arise in the micropolar
medium it is convenient to expand the displacement and rotation vectors
in terms of scalar and vector potentials

.. math::

  \begin{align}
  \mathbf{u} &= \nabla \phi + \nabla\times\boldsymbol{\Gamma}\, ,\\
  \boldsymbol{\theta} &= \nabla \tau + \nabla\times\mathbf{E}\, ,
  \end{align}

subject to the conditions:

.. math::

  \begin{align}
  &\nabla\cdot\boldsymbol{\Gamma} = 0\\
  &\nabla\cdot\mathbf{E} = 0\, .
  \end{align}

Using the above in the displacements equations of motion yields the
following equations, after some manipulations

.. math::

  \begin{align}
  c_1^2 \nabla^2 \phi &= \frac{\partial^2 \phi}{\partial t^2}\, ,\\
  c_3^2 \nabla^2 \tau - 2Q^2\tau &= \frac{\partial^2 \tau}{\partial t^2}\, ,\\
  \begin{bmatrix}
  c_2^2 \nabla^2 &K^2\nabla\times\, ,\\
  Q^2\nabla\times &c_4^2\nabla^2 - 2Q^2
  \end{bmatrix}
  \begin{Bmatrix} \boldsymbol{\Gamma}\\ \mathbf{E}\end{Bmatrix} &=
  \frac{\partial^2}{\partial t^2} \begin{Bmatrix} \boldsymbol{\Gamma}\\ \mathbf{E}\end{Bmatrix} \, ,
  \end{align}

where we can see that the equations for the scalar potentials are
uncoupled, while the ones for the vector potentials are coupled.

Writing the vector potentials as plane waves of amplitude
:math:`\mathbf{A}` and :math:`\mathbf{B}`, wave number
:math:`\kappa` and circular frequency :math:`\omega` that propagate
along the (x) axis,

.. math::

  \begin{align}
  \boldsymbol{\Gamma} &= \mathbf{A}\exp(i\kappa x - i\omega t)\\
  \mathbf{E} &= \mathbf{B}\exp(i\kappa x - i\omega t)\, .
  \end{align}

We can do these calculations using some the functions available
functions in the package.

.. code:: python

    from sympy import Matrix, diff, symbols, exp, I, sqrt
    from sympy import simplify, expand, solve, limit
    from sympy import init_printing, pprint, factor
    from continuum_mechanics.vector import lap_vec, curl, div

.. code:: python

    A1, A2, A3, B1, B2, B3 = symbols("A1 A2 A3 B1 B2 B3")
    kappa, omega, t, x = symbols("kappa omega t x")
    c1, c2, c3, c4, K, Q = symbols("c1 c2 c3 c4 K Q", positive=True)

We define the vector potentials :math:`\boldsymbol{\Gamma}` and
:math:`\mathbf{E}`.

.. code:: python

    Gamma = Matrix([A1, A2, A3]) * exp(I*kappa*x - I*omega*t)
    E = Matrix([B1, B2, B3]) * exp(I*kappa*x - I*omega*t)

And compute the equations using the vector operators. Namely, the
Laplace (:py:func:`vector.lap_vec`)  and curl (:py:func:`vector.curl`)
operators.

.. code:: python

    eq1 = c2**2 * lap_vec(Gamma) + K**2*curl(E) - Gamma.diff(t, 2)
    eq2 = Q**2 * curl(Gamma) + c4**2*lap_vec(E) - 2*Q**2*E - E.diff(t, 2)
    eq1 = simplify(eq1/exp(I*kappa*x - I*omega*t))
    eq2 = simplify(eq2/exp(I*kappa*x  - I*omega*t))
    eq = eq1.col_join(eq2)

We can compute the matrix for the system using
`.jacobian() <https://docs.sympy.org/1.5.1/modules/matrices/matrices.html#sympy.matrices.matrices.MatrixCalculus.jacobian>`__

.. code:: python

    M = eq.jacobian([A1, A2, A3, B1, B2, B3])
    M




.. math::

    \left[\begin{matrix}- c_{2}^{2} \kappa^{2} + \omega^{2} & 0 & 0 & 0 & 0 & 0\\0 & - c_{2}^{2} \kappa^{2} + \omega^{2} & 0 & 0 & 0 & - i K^{2} \kappa\\0 & 0 & - c_{2}^{2} \kappa^{2} + \omega^{2} & 0 & i K^{2} \kappa & 0\\0 & 0 & 0 & - 2 Q^{2} - c_{4}^{2} \kappa^{2} + \omega^{2} & 0 & 0\\0 & 0 & - i Q^{2} \kappa & 0 & - 2 Q^{2} - c_{4}^{2} \kappa^{2} + \omega^{2} & 0\\0 & i Q^{2} \kappa & 0 & 0 & 0 & - 2 Q^{2} - c_{4}^{2} \kappa^{2} + \omega^{2}\end{matrix}\right]



And, we are interested in the determinant of the matrix :math:`M`.

.. code:: python

    factor(M.det())




.. math::

    \left(c_{2} \kappa - \omega\right) \left(c_{2} \kappa + \omega\right) \left(2 Q^{2} + c_{4}^{2} \kappa^{2} - \omega^{2}\right) \left(- K^{2} Q^{2} \kappa^{2} + 2 Q^{2} c_{2}^{2} \kappa^{2} - 2 Q^{2} \omega^{2} + c_{2}^{2} c_{4}^{2} \kappa^{4} - c_{2}^{2} \kappa^{2} \omega^{2} - c_{4}^{2} \kappa^{2} \omega^{2} + \omega^{4}\right)^{2}



The roots for this polynomial (in :math:`\omega^2`) represent the
dispersion relations.

.. code:: python

    disps = solve(M.det(), omega**2)
    for disp in disps:
        display(disp)



.. math::

    c_{2}^{2} \kappa^{2}



.. math::

    2 Q^{2} + c_{4}^{2} \kappa^{2}



.. math::

    Q^{2} + \frac{c_{2}^{2} \kappa^{2}}{2} + \frac{c_{4}^{2} \kappa^{2}}{2} - \frac{1}{2} \sqrt{4 K^{2} Q^{2} \kappa^{2} + 4 Q^{4} - 4 Q^{2} c_{2}^{2} \kappa^{2} + 4 Q^{2} c_{4}^{2} \kappa^{2} + c_{2}^{4} \kappa^{4} - 2 c_{2}^{2} c_{4}^{2} \kappa^{4} + c_{4}^{4} \kappa^{4}}



.. math::

    Q^{2} + \frac{c_{2}^{2} \kappa^{2}}{2} + \frac{c_{4}^{2} \kappa^{2}}{2} + \frac{1}{2} \sqrt{4 K^{2} Q^{2} \kappa^{2} + 4 Q^{4} - 4 Q^{2} c_{2}^{2} \kappa^{2} + 4 Q^{2} c_{4}^{2} \kappa^{2} + c_{2}^{4} \kappa^{4} - 2 c_{2}^{2} c_{4}^{2} \kappa^{4} + c_{4}^{4} \kappa^{4}}


References
----------

.. [NOWACKI]
    Nowacki, W. (1986). Theory of asymmetric elasticity.
    pergamon Press, Headington Hill Hall, Oxford OX 3 0 BW, UK, 1986.

.. [GUARIN]
    Guarín-Zapata, N., Gomez, J., Valencia, C., Dargush, G. F., &
    Hadjesfandiari, A. R. (2020). Finite element modeling of
    micropolar-based phononic crystals. Wave Motion, 92, 102406.
