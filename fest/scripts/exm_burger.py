


from firedrake import *
from firedrake.pyplot import tripcolor
import matplotlib.pyplot as plt



if __name__ == '__main__':
    m = UnitIntervalMesh(30)
    mesh = ExtrudedMesh(m, layers=4, layer_height=0.2)

    # Implement the solution space, where it uses CG for both spatial and time element

    W_s = FiniteElement("CG", "interval", 1)   # spatial element
    W_t = FiniteElement("CG", "interval", 1)   # time element
    W_elt = TensorProductElement(W_s, W_t)    
    U = FunctionSpace(mesh, W_elt) 

    # Implement the test space, where it uses CG for spatial and DG for time with 1 degree less
    V_s = FiniteElement("CG", "interval", 1)  
    V_t = FiniteElement("DG", "interval", 0) 
    V_el =  TensorProductElement(V_s, V_t)  
    V = FunctionSpace(mesh, V_el)  

    # Restrict BC to bottom of solution space
    U_res = RestrictedFunctionSpace(U, boundary_set=['bottom']) 
    U_lgmap = U_res.topological.local_to_global_map(None)


    V_res = RestrictedFunctionSpace(V, boundary_set=['bottom']) # this changes nothing since DG elelments don't have node on boundary.
    V_lgmap = V_res.topological.local_to_global_map(None)



    u = Function(U) # for non-lineaer problem, need to directly define a function instead of trialfunction and directly solve for the function itself
                        # due to the difference in approaches dealing with non-linear system.


    v = TestFunction(V_res)


    x, t = SpatialCoordinate(mesh)


    # Forming the boundary condition as cos(2pi * x), which is essentially the initial time condition
    u_init = Function(U_res)
    u_init.interpolate(sin(pi* x))

    bc = DirichletBC(U_res, u_init, 'bottom')

    sol = Function(U)

    nu = 0.0001

    # LHS for Burger equation

    F = (u.dx(1)*v + u*u.dx(0)*v - nu*u.dx(0).dx(0)*v)*dx

    solve(F == 0, u, bcs=[bc], restrict=True)


    tripcolor(u)
    plt.show()