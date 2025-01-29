"""This file includes a function version of the construction of the immersed mesh"""

from firedrake import *
from firedrake.pyplot import tripcolor, plot 
import matplotlib.pyplot as plt


# The exact same set up as in the heat equation 

m = UnitIntervalMesh(5)
mesh = ExtrudedMesh(m, layers=10, layer_height=10)

# Implement the solution space, where it uses CG for both spatial and time element

W_s = FiniteElement("CG", "interval", 3)   # spatial element
W_t = FiniteElement("CG", "interval", 3)   # time element
W_elt = TensorProductElement(W_s, W_t)    
U = FunctionSpace(mesh, W_elt) 

# Implement the test space, where it uses CG for spatial and DG for time with 1 degree less
V_s = FiniteElement("CG", "interval", 3)  
V_t = FiniteElement("DG", "interval", 2) 
V_el =  TensorProductElement(V_s, V_t)  
V = FunctionSpace(mesh, V_el)  

# Restrict BC to bottom of solution space
U_res = RestrictedFunctionSpace(U, boundary_set=['bottom']) 
U_lgmap = U_res.topological.local_to_global_map(None)


V_res = RestrictedFunctionSpace(V, boundary_set=['bottom']) # this changes nothing since DG elelments don't have node on boundary.
V_lgmap = V_res.topological.local_to_global_map(None)


u = TrialFunction(U)

v = TestFunction(V)

u_init = Function(U_res)


x, t = SpatialCoordinate(mesh)

# Forming the boundary condition as cos(2pi * x), which is essentially the initial time condition
u_init.interpolate(cos(2*pi*x))

# bc = DirichletBC(U_res, cos(pi*x), 'bottom')
bc = DirichletBC(U_res, u_init, 'bottom')

sol1 = Function(U)

mu = 0.001
# LHS for Linear Advection equation
a = (u.dx(1) * v - u * mu * v.dx(0)) * dx



# LHS for Heat Equation
h = (u.dx(1) * v  + u.dx(0) * mu * v.dx(0)) * dx


L = Constant(0.0) * v * dx

solve(h == L, sol1, bcs=[bc], restrict=True)


def immersed_mesh(m, W_s, sol1, pos):
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
        x_f.interpolate(as_vector([x, 20]))
    else:
        raise NotImplementedError

    # Create the immersed mesh (1D mesh in 2D space)
    m_imm = Mesh(x_f)
    
    # Define the function space on the immersed mesh and interpolate the solution
    UFs_imm = FunctionSpace(m_imm, W_s)
    u_f = Function(UFs_imm)
    if pos == 'top':
        u_f.interpolate(sol1)
        # Define the function space on the original mesh
        UFs_1D = FunctionSpace(m, W_s)
        u_1d = Function(UFs_1D)
        
        u_1d.dat.data_wo[:] = u_f.dat.data_ro

        return u_1d
    else:
        u_f.dat.data_wo[:] = sol1.dat.data_ro
        return u_f

    

u_t = immersed_mesh(m, W_s, sol1, 'top')

u_b = immersed_mesh(m, W_s, u_t, 'bottom')


u_2d = Function(U_res)

u_2d.interpolate(u_b, allow_missing_dofs=True)

bc_renew = DirichletBC(U_res, u_2d, 'bottom')

sol2 = Function(U_res)

solve(h == L, sol2, bcs=[bc_renew], restrict=True)

print(norm(sol2))

#tripcolor(sol2)
#plt.show()

