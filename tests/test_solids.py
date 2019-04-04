# -*- coding: utf-8 -*-
"""
Test for solids at continuum_mechanics package

"""
from __future__ import division, print_function
import sympy as sym
from sympy import symbols, sin, cos, Abs
from sympy import Matrix, simplify
from continuum_mechanics.solids import navier_cauchy, c_cst

x, y, z = sym.symbols("x y z")


def test_navier_op():

    # Rotating cylinder in polar coordinates
    r, rho, Omega, E, nu, R = symbols("r rho Omega E nu R")
    coords = (r, y, z)
    h_vec = (1, r, 1)
    u = [rho*Omega**2*(1 + nu)*(1 - 2*nu)/(8*E*(1 - nu)) *r* ((3 - 2*nu)*R**2 - r**2),
         0, 0]
    lamda = E*nu/((1 + nu)*(1 - 2*nu))
    mu = E/(2*(1 + nu))
    params = lamda, mu
    b = navier_cauchy(u, params, coords, h_vec)
    b_anal = Matrix([
            [-Omega**2*r*rho],
            [              0],
            [              0]])
    assert b == b_anal


def test_c_cst():

    # Rotating cylinder in polar coordinates
    r, rho, Omega, E, nu, R, eta = symbols("r rho Omega E nu R eta")
    coords = (r, y, z)
    h_vec = (1, r, 1)
    u = [rho*Omega**2*(1 + nu)*(1 - 2*nu)/(8*E*(1 - nu)) *r* ((3 - 2*nu)*R**2 - r**2),
         0, 0]
    lamda = E*nu/((1 + nu)*(1 - 2*nu))
    mu = E/(2*(1 + nu))
    params = lamda, mu, eta
    b = c_cst(u, params, coords, h_vec)
    b_anal = Matrix([
            [-Omega**2*r*rho],
            [              0],
            [              0]])
    assert b == b_anal
