# -*- coding: utf-8 -*-
"""
Test for vector at continuum_mechanics package

"""
from __future__ import division, print_function
import sympy as sym
from sympy import symbols, sin, cos, Abs
from sympy import Matrix, Function, diff
from continuum_mechanics.vector import (scale_coeff, levi_civita, dual_tensor,
                                        dual_vector, grad, grad_vec, div,
                                        div_tensor, curl, lap, lap_vec)

x, y, z = sym.symbols("x y z")

#%% Curvilinear coordinates
def test_scale_coeff():
    r, theta, phi = sym.symbols("r theta phi", positive=True)
    r_vec = [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]
    coords = [r, theta, phi]
    h_vec = scale_coeff(r_vec, coords)
    assert h_vec == (1, r, r*Abs(sin(theta)))


def unit_vec_deriv():
    pass

#%% Vector analysis
def test_levi_civita():
    assert levi_civita(1, 2, 3) == 1
    assert levi_civita(1, 3, 2) == -1
    assert levi_civita(1, 1, 2) == 0


def test_dual_tensor():
    vector = Matrix([1, 2, 3])
    tensor = Matrix([
        [0, 3, -2],
        [-3, 0, 1],
        [2, -1, 0]])
    assert dual_tensor(vector) == tensor
    assert dual_vector(tensor) == vector


def test_dual_vector():
    tensor = Matrix([
        [0, 3, -2],
        [-3, 0, 1],
        [2, -1, 0]])
    vector = Matrix([1, 2, 3])
    assert dual_vector(tensor) == vector
    assert dual_tensor(vector) == tensor


#%% Differential operators
def test_grad():
    gradient = grad(-(cos(x)**2 + cos(y)**2)**2)
    expected_gradient = Matrix([
            [4*(cos(x)**2 + cos(y)**2)*sin(x)*cos(x)],
            [4*(cos(x)**2 + cos(y)**2)*sin(y)*cos(y)],
            [0]])
    assert gradient.equals(expected_gradient)


def test_grad_vec():
    # Cartesian coordinates
    gradient = grad_vec([x*y*z, x*y*z, x*y*z])
    expected_gradient = Matrix([
            [y*z, x*z, x*y],
            [y*z, x*z, x*y],
            [y*z, x*z, x*y]])
    assert gradient.equals(expected_gradient.T)

    # Spherical coordinates
    r, phi, theta = symbols("r phi theta")
    Ar, Ap, At = symbols("A_r A_phi A_theta", cls=Function)
    A1 = Ar(r, phi, theta)
    A2 = Ap(r, phi, theta)
    A3 = At(r, phi, theta)
    A = Matrix([A1, A2, A3])
    gradient = grad_vec(A, (r, phi, theta), (1, r, r*sin(phi)))
    expected_gradient = Matrix([
        [diff(A1, r), diff(A2, r), diff(A3, r)],
        [(diff(A1, phi) - A2)/r, (A1 + diff(A2, phi))/r,
         diff(A3, phi)/r],
        [diff(A1,theta)/(r*sin(phi)) - A3/r,
         (diff(A2, theta) - A3*cos(phi))/(r*sin(phi)),
         (A1*sin(phi) + A2*cos(phi) + diff(A3, theta))/(r*sin(phi))]])
    assert gradient.equals(expected_gradient)


def test_div():
    divergence = div([x**2 + y*z, y**2 + x*z, z**2 + x*y])
    expected_divergence = 2*x + 2*y + 2*z
    assert divergence == expected_divergence


def test_div_tensor():

    # Cylindrical coordinates for symmetric tensor
    r, phi = symbols("r phi")
    A11, A22, A33 = symbols("A_11 A_22 A_33", cls=Function)
    A12, A13, A23 = symbols("A_12 A_13 A_23", cls=Function)
    A = Matrix([
        [A11(r, phi, z), A12(r, phi, z), A13(r, phi, z)],
        [A12(r, phi, z), A22(r, phi, z), A23(r, phi, z)],
        [A13(r, phi, z), A23(r, phi, z), A33(r, phi, z)]])
    divergence = div_tensor(A, (r, phi, z), (1, r, 1))
    div1 = diff(A11(r, phi, z), r) + 1/r*diff(A12(r, phi, z), phi) \
         + diff(A13(r, phi, z), z) + (A11(r, phi, z) - A22(r, phi, z))/r
    div2 = diff(A12(r, phi, z), r) + 1/r*diff(A22(r, phi, z), phi) \
         + diff(A23(r, phi, z), z) + 2*A12(r, phi, z)/r
    div3 = diff(A13(r, phi, z), r) + 1/r*diff(A23(r, phi, z), phi) \
         + diff(A33(r, phi, z), z) + A13(r, phi, z)/r
    expected_div = Matrix([div1, div2, div3])
    assert divergence.equals(expected_div)



def test_curl():
    rot = curl([0, -x**2, 0])
    expected_rot = Matrix([[0], [0], [-2*x]])
    assert rot == expected_rot


def test_lap():
    laplacian = lap(x**2 +  y**2 + z**2)
    expected_laplacian = 6
    assert laplacian == expected_laplacian


def test_lap_vec():
    laplacian = lap_vec([x**2, y**2, z**2])
    expected_laplacian = Matrix([[2], [2], [2]])
    assert laplacian == expected_laplacian
