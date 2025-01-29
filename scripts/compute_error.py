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

n_layers = 2

# The exact same set up as in the heat equation 

m = UnitIntervalMesh(4)
mesh_full = ExtrudedMesh(m, layers=n_layers, layer_height=0.5)
mesh_half = ExtrudedMesh(m, layers=n_layers//2, layer_height=0.25)

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

mu = 0.001
# LHS for Linear Advection equation
# a = (u.dx(1) * v - u * mu * v.dx(0)) * dx


# LHS for Heat Equation
h_1 = (u_1.dx(1) * v_1  + u_1.dx(0) * mu * v_1.dx(0)) * dx
h_2 = (u_2.dx(1) * v_2  + u_2.dx(0) * mu * v_2.dx(0)) * dx


L_1 = Constant(0.0) * v_1 * dx
L_2 = Constant(0.0) * v_2 * dx

solve(h_1 == L_1, sol_1, bcs=[bc_1], restrict=True)
solve(h_2 == L_2, sol_2_1, bcs=[bc_2], restrict=True)





def immersed_mesh(m, W_s, sol1, pos, layer_height=0.5):
    """
    Perform the extract-reinsert algorithm using immersed mesh

    param m: the original 1D mesh (unextruded dimension)
    param W_s: the spatial finite element (FE associated with the original mesh)
    param sol1: the function being extracted / reinserted
    param pos: a string being either 'top' or 'bottom'

    return u_1d: a 1d function being extracted out
    return u_f: a 2D function being re-inserted into the mesh
    """
    # Create the function space for the top of the extrusion
    Fs_imm = VectorFunctionSpace(m, 'CG', 1, dim=2)  
    x_f = Function(Fs_imm)  # Create the storer function in Fs_top

    # Get the spatial coordinate
    x, = SpatialCoordinate(m)
    
    # Interpolate the storer with information of the pos
    if pos == 'bottom':
        x_f.interpolate(as_vector([x, 0]))
    elif pos == 'top':
        x_f.interpolate(as_vector([x, layer_height]))
    else:
        raise NotImplementedError

    # Create the immersed mesh (1D mesh in 2D space)
    m_imm = Mesh(x_f)
    # Define the function space on the immersed mesh and interpolate the solution
    UFs_imm = FunctionSpace(m_imm, W_s)
    u_f = Function(UFs_imm)
    if pos == 'top':
        u_f.interpolate(sol1, allow_missing_dofs=True)
        # Define the function space on the original mesh
        UFs_1D = FunctionSpace(m, W_s)
        u_1d = Function(UFs_1D)
        
        u_1d.dat.data_wo[:] = u_f.dat.data_ro

        return u_1d
    else:
        u_f.dat.data_wo[:] = sol1.dat.data_ro
        return u_f

    

u_t = immersed_mesh(m, W_s, sol_2_1, 'top', layer_height=0.25)

u_b = immersed_mesh(m, W_s, u_t, 'bottom')


u_2d = Function(U_res_half)

u_2d.interpolate(u_b, allow_missing_dofs=True)

bc_renew = DirichletBC(U_res_half, u_2d, 'bottom')

sol_2_2 = Function(U_res_half)

solve(h_2 == L_2, sol_2_2, bcs=[bc_renew], restrict=True)



res_full = immersed_mesh(m, W_s, sol_1, 'top', layer_height=0.5)
res_half = immersed_mesh(m, W_s, sol_2_2, 'top', layer_height=0.25)

print(errornorm(res_full, res_half))
