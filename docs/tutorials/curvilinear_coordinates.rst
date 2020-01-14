======================================
Introduction to orthogonal coordinates
======================================

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/nicoguaro/continuum_mechanics/master?filepath=docs%2Ftutorials%2Fcurvilinear_coordinates.ipynb

In :math:`\mathbb{R}^3`, we can think that each point is given by the
intersection of three surfaces. Thus, we have three families of curved
surfaces that intersect each other at right angles. These surfaces are
orthogonal locally, but not (necessarily) globally, and are defined by

.. math:: u_1 = f_1(x, y, z)\, ,\quad u_2 = f_2(x, y, z)\, ,\quad u_3=f_3(x, y, z) \, .

These functions should be invertible, at least locally, and we can also
write

.. math:: x = x(u_1, u_2, u_3)\, ,\quad y = y(u_1, u_2, u_3)\, ,\quad z = z(u_1, u_2, u_3)\, ,

where :math:`x, y, z` are the usual Cartesian coordinates. The curve
defined by the intersection of two of the surfaces gives us one of the
coordinate curves.

Scale factors
-------------

Since we are interested in how these surface intersect each other
locally, we want to express differential vectors in terms of the
coordinates. Thus, the differential for the position vector
(:math:`\mathbf{r}`) is given by

.. math::

   \mathrm{d}\mathbf{r} = \frac{\partial\mathbf{r}}{\partial u_1}\mathrm{d}u_1
   + \frac{\partial\mathbf{r}}{\partial u_2}\mathrm{d}u_2
   + \frac{\partial\mathbf{r}}{\partial u_3}\mathrm{d}u_3\, ,

or

.. math:: \mathrm{d}\mathbf{r} = \sum_{i=1}^3 \frac{\partial\mathbf{r}}{\partial u_i}\mathrm{d}u_i\, .

The factor :math:`\partial \mathbf{r}/\partial u_i` is a non-unitary
vector that takes into account the variation of :math:`\mathbf{r}` in
the direction of :math:`u_i`, and is then tangent to the coordinate
curve :math:`u_i`. We can define a normalized basis
:math:`\hat{\mathbf{e}}_i` using

.. math:: \frac{\partial\mathbf{r}}{\partial u_i} = h_i \hat{\mathbf{e}}_i\, .

The coefficients :math:`h_i` are functions of :math:`u_i` and we call
them *scale factors*. They are really important since they allow us to
*measure* distances while we move along our coordinates. We would need
them to define vector operators in orthogonal coordinates. When the
coordinates are not orthogonal we would need to use the `metric
tensor <https://en.wikipedia.org/wiki/Metric_tensor>`__, but we are
going to restrict ourselves to orthogonal systems.

Hence, we have the following

.. math::

   \begin{align}
   &h_i = \left|\frac{\partial\mathbf{r}}{\partial u_i}\right|\, ,\\
   &\hat{\mathbf{e}}_i = \frac{1}{h_i} \frac{\partial \mathbf{r}}{\partial u_i}\, .
   \end{align}

Curvilinear coordinates available
---------------------------------

The following coordinate systems are available:

-  Cartesian;

-  Cylindrical;

-  Spherical;

-  Parabolic cylindrical;

-  Parabolic;

-  Paraboloidal;

-  Elliptic cylindrical;

-  Oblate spheroidal;

-  Prolate spheroidal;

-  Ellipsoidal;

-  Bipolar cylindrical;

-  Toroidal;

-  Bispherical; and

-  Conical.

To obtain the transformation for a given coordinate system we can use
the function :py:func:`vector.transform_coords`.

.. code:: python

    import sympy as sym
    from continuum_mechanics import vector

First, we define the variables for the coordinates :math:`(u, v, w)`.

.. code:: python

    sym.init_printing()
    u, v, w = sym.symbols("u v w")

And, we compute the coordinates for the **parabolic** system using
:py:func:`vector.transform_coords`. The first parameter is a string
defining the coordinate system and the second is a tuple with the
coordinates.

.. code:: python

    vector.transform_coords("parabolic", (u, v, w))




.. math::

    \displaystyle \left( u v \cos{\left(w \right)}, \  u v \sin{\left(w \right)}, \  \frac{u^{2}}{2} - \frac{v^{2}}{2}\right)



The scale factors for the coordinate systems mentioned above are
availabe. We can compute them for bipolar cylindrical coordinates. The
coordinates are defined by

.. math::

   \begin{align}
   &x = a \frac{\sinh\tau}{\cosh\tau - \cos\sigma}\, ,\\
   &y = a \frac{\sin\sigma}{\cosh\tau - \cos\sigma}\, ,\\
   &z = z\, ,
   \end{align}

and have the following scale factors

.. math:: h_\sigma = h_\tau = \frac{a}{\cosh\tau - \cos\sigma}\, ,

and :math:`h_z = 1`.

.. code:: python

    sigma, tau, z, a = sym.symbols("sigma tau z a")
    z = sym.symbols("z")
    scale = vector.scale_coeff_coords("bipolar_cylindrical", (sigma, tau, z), a=a)
    scale




.. math::

    \displaystyle \left( \frac{a}{- \cos{\left(\sigma \right)} + \cosh{\left(\tau \right)}}, \  \frac{a}{- \cos{\left(\sigma \right)} + \cosh{\left(\tau \right)}}, \  1\right)



Finally, we can compute vector operators for different coordinates.

The Laplace operator for the bipolar cylindrical system is given by

.. math::


   \nabla^2 \phi =
   \frac{1}{a^2} \left( \cosh \tau - \cos\sigma \right)^{2}
   \left(
   \frac{\partial^2 \phi}{\partial \sigma^2} +
   \frac{\partial^2 \phi}{\partial \tau^2}
   \right) +
   \frac{\partial^2 \phi}{\partial z^2}\, ,

and we can compute it using the function :py:func:`vector.lap`. For this function, the
first parameter is the expression that we want to compute the Laplacian
for, the second parameter is a tuple with the coordinates and the third
parameter is a tuple with the scale factors.

.. code:: python

    phi = sym.symbols("phi", cls=sym.Function)
    lap = vector.lap(phi(sigma, tau, z), coords=(sigma, tau, z), h_vec=scale)
    sym.simplify(lap)




.. math::

    \displaystyle \frac{a^{2} \frac{\partial^{2}}{\partial z^{2}} \phi{\left(\sigma,\tau,z \right)} + \left(\cos{\left(\sigma \right)} - \cosh{\left(\tau \right)}\right)^{2} \frac{\partial^{2}}{\partial \sigma^{2}} \phi{\left(\sigma,\tau,z \right)} + \left(\cos{\left(\sigma \right)} - \cosh{\left(\tau \right)}\right)^{2} \frac{\partial^{2}}{\partial \tau^{2}} \phi{\left(\sigma,\tau,z \right)}}{a^{2}}
