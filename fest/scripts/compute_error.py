"""
This script is for calculating the error between solving the space-time 
system in once comparing to iteratively solving through the time steps, which
should theoretically be up to machine error (1e-16 for Float16), as the 
numerical operations involved are essentially the same
"""

"""This file includes a function version of the construction of the immersed mesh"""

from firedrake import *
from firedrake.pyplot import tripcolor, plot
import matplotlib.pyplot as plt
from fest import transfer

n_layers = 2

# The exact same set up as in the heat equation 

m = UnitIntervalMesh(100)
mesh_full = ExtrudedMesh(m, layers=2, layer_height=0.5)
mesh_half = ExtrudedMesh(m, layers=1, layer_height=0.25)

# Implement the solution space, where it uses CG for both spatial and time element

W_s = FiniteElement("CG", "interval", 3)   # spatial element
W_t = FiniteElement("CG", "interval", 3)   # time element
W_elt = TensorProductElement(W_s, W_t)    
U_full = FunctionSpace(mesh_full, W_elt) 

W_s_h = FiniteElement("CG", "interval", 3)   # spatial element
W_t_h = FiniteElement("CG", "interval", 3)   # time element
W_elt_h = TensorProductElement(W_s_h, W_t_h)  
U_half = FunctionSpace(mesh_half, W_elt_h)

# Implement the test space, where it uses CG for spatial and DG for time with 1 degree less
V_s = FiniteElement("CG", "interval", 3)
V_t = FiniteElement("DG", "interval", 2)
V_elt =  TensorProductElement(V_s, V_t)
V_full = FunctionSpace(mesh_full, V_elt)

V_s_h = FiniteElement("CG", "interval", 3)
V_t_h = FiniteElement("DG", "interval", 2)
V_elt_h =  TensorProductElement(V_s, V_t)
V_half = FunctionSpace(mesh_half, V_elt_h)

# Restrict BC to bottom of solution space
U_res_full = RestrictedFunctionSpace(U_full, boundary_set=['bottom']) 
U_lgmap = U_res_full.topological.local_to_global_map(None)
U_res_half = RestrictedFunctionSpace(U_half, boundary_set=['bottom'])


V_res_full = RestrictedFunctionSpace(V_full, boundary_set=['bottom']) # this changes nothing since DG elelments don't have node on boundary.
V_lgmap = V_res_full.topological.local_to_global_map(None)
V_res_half = RestrictedFunctionSpace(V_half, boundary_set=['bottom'])


u_1 = TrialFunction(U_full)
u_2 = TrialFunction(U_half)

v_1 = TestFunction(V_full)
v_2 = TestFunction(V_half)

u_init_1 = Function(U_res_full)
u_init_2 = Function(U_res_half)

x, = SpatialCoordinate(m)
x_1, t_1 = SpatialCoordinate(mesh_full)
x_2, t_2 = SpatialCoordinate(mesh_half)

# Forming the boundary condition as cos(2pi * x), which is essentially the initial time condition
u_init_1.interpolate(cos(2*pi*x_1))
u_init_2.interpolate(cos(2*pi*x_2))

# bc = DirichletBC(U_res, cos(pi*x), 'bottom')
bc_1 = DirichletBC(U_res_full, u_init_1, 'bottom')
bc_2 = DirichletBC(U_res_half, u_init_2, 'bottom')

sol_1 = Function(U_full)
sol_2_1 = Function(U_half)

mu = 1e-5
# LHS for Linear Advection equation
# a = (u.dx(1) * v - u * mu * v.dx(0)) * dx


# LHS for Heat Equation
h_1 = (u_1.dx(1) * v_1  + u_1.dx(0) * mu * v_1.dx(0)) * dx
h_2 = (u_2.dx(1) * v_2  + u_2.dx(0) * mu * v_2.dx(0)) * dx


L_1 = Constant(0.0) * v_1 * dx
L_2 = Constant(0.0) * v_2 * dx

solve(h_1 == L_1, sol_1, bcs=[bc_1], restrict=True)
solve(h_2 == L_2, sol_2_1, bcs=[bc_2], restrict=True)



u_t = transfer(m, W_s, sol_2_1, 'top', layer_height=0.25)

u_b = transfer(m, W_s, u_t, 'bottom')


u_2d = Function(U_res_half)

u_2d.interpolate(u_b, allow_missing_dofs=True)

bc_renew = DirichletBC(U_res_half, u_2d, 'bottom')

sol_2_2 = Function(U_half)

solve(h_2 == L_2, sol_2_2, bcs=[bc_renew], restrict=True)



res_full = transfer(m, W_s, sol_1, 'top', layer_height=0.5)
res_half = transfer(m, W_s, sol_2_2, 'top', layer_height=0.25)



print(errornorm(res_full, res_half)) # compare the solution of two numerical methods

# the analytical solution to the heaet equation with the given bc is cos(2*pi*x)*exp(-4*(pi**2)*mu*t)

print(errornorm(cos(2*pi*x)*exp(-4*(pi**2)*mu*0.5), res_full))
