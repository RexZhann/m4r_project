from firedrake import *
from firedrake.pyplot import tripcolor, plot
import matplotlib.pyplot as plt


def construct_scheme(mesh_size, ex_num_layers, ex_layer_h, deg, bc_expr):
    m = UnitIntervalMesh(mesh_size)
    mesh = ExtrudedMesh(m, layers=ex_num_layers, layer_height=ex_layer_h)

    # Implement the solution space, where it uses CG for both spatial and time element

    W_s = FiniteElement("CG", "interval", deg)   # spatial element
    W_t = FiniteElement("CG", "interval", deg)   # time element
    W_elt = TensorProductElement(W_s, W_t)    
    U = FunctionSpace(mesh, W_elt) 

    # Implement the test space, where it uses CG for spatial and DG for time with 1 degree less
    V_s = FiniteElement("CG", "interval", deg)  
    V_t = FiniteElement("DG", "interval", deg - 1) 
    V_el =  TensorProductElement(V_s, V_t)  
    V = FunctionSpace(mesh, V_el)  

    # Restrict BC to bottom of solution space
    U_res = RestrictedFunctionSpace(U, boundary_set=['bottom']) 
    # U_lgmap = U_res.topological.local_to_global_map(None)


    V_res = RestrictedFunctionSpace(V, boundary_set=['bottom']) # this changes nothing since DG elelments don't have node on boundary.
    # V_lgmap = V_res.topological.local_to_global_map(None)


    u = TrialFunction(U)

    v = TestFunction(V)

    u_init = Function(U_res)


    x, t = SpatialCoordinate(mesh)

    # Forming the boundary condition as cos(2pi * x), which is essentially the initial time condition
    u_init.interpolate(bc_expr)

    # bc = DirichletBC(U_res, cos(pi*x), 'bottom')
    bc = DirichletBC(U_res, u_init, 'bottom')

    sol1 = Function(U)

    return bc, sol1