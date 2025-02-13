from firedrake import *
from firedrake.pyplot import tripcolor, plot
import matplotlib.pyplot as plt


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

def get_spacetime_errornorm(sp_res, t_res, t_end, deg, bc_expr, h, L, exact, num_iter):
    '''
    param sp_res: the number of element on spatial dimensions
    param t_res: the number of cells on time dimensions
    param t_end: the final time of evaluation
    param deg: the polynomial degree of each element
    param bc_expr: an UFL expression for boundary condition
    param h, L: the governing equation of the system
    param exact: UFL expression of the exact solution
    param num_iter: numbers of iteration, associated with t_res

    return err: the error norm between numerical method and exact solution
    '''
    m = UnitIntervalMesh(sp_res)
    mesh_ex= ExtrudedMesh(m, layers=t_res, layer_height=t_end/num_iter)

    W_s = FiniteElement("CG", "interval", deg)   # spatial element
    W_t = FiniteElement("CG", "interval", deg)   # time element
    W_elt = TensorProductElement(W_s, W_t)    
    U = FunctionSpace(mesh_ex, W_elt) 

    V_s = FiniteElement("CG", "interval", deg)
    V_t = FiniteElement("DG", "interval", deg - 1)
    V_elt =  TensorProductElement(V_s, V_t)
    V = FunctionSpace(mesh_ex, V_elt)

    U_res = RestrictedFunctionSpace(U, boundary_set=['bottom'])
    V_res = RestrictedFunctionSpace(V, boundary_set=['bottom'])

    u = TrialFunction(U)
    v = TestFunction(V)

    u_init = Function(U_res)

    x, = SpatialCoordinate(m)
    x_ex, t = SpatialCoordinate(mesh_ex)

    u_init.interpolate(bc_expr(x_ex))
    bc = DirichletBC(U_res, u_init, 'bottom')

    sol_0 = Function(U)
    

    sol = sol_0
    bc_renew = bc
    for _ in range(num_iter):

        solve(h(u, v) == L(u, v), sol, bcs=[bc_renew], restrict=True)
        u_t = immersed_mesh(m, W_s, sol, 'top', layer_height=t_end/num_iter)
        u_b = immersed_mesh(m, W_s, u_t, 'bottom')

        u_2d = Function(U_res)
        u_2d.interpolate(u_b, allow_missing_dofs=True)
        bc_renew = DirichletBC(U_res, u_2d, 'bottom')
    
    err = errornorm(exact(x, t_end), u_t)

    return err

mu = 0.00001

def bc_expr(x):
    return cos(2*pi*x)

def h(u, v, mu=mu):
    return (u.dx(1) * v + u.dx(0) * mu * v.dx(0)) * dx

def L(u, v):
    return Constant(0.0) * v * dx

def exact(x, t):
    return cos(2*pi*x)*exp(-4*(pi**2) * mu * t)




'''sp_list = [10, 50, 100, 500, 1000, 5000]

error_list_full_sp = [get_spacetime_errornorm(sp, 2, 0.5, 3, bc_expr, h, L, exact, 1) for sp in sp_list]
error_list_step_sp = [get_spacetime_errornorm(sp, 1, 0.25, 3, bc_expr, h, L, exact, 2) for sp in sp_list]

'''

t_list = [2, 4, 6, 8, 10, 12]

error_list_full_t = [get_spacetime_errornorm(50, t, 0.5, 3, bc_expr, h, L, exact, 1) for t in t_list]
error_list_step_t = [get_spacetime_errornorm(50, 1, 0.5, 3, bc_expr, h, L, exact, t) for t in t_list]

plt.loglog(error_list_full_t, error_list_step_t)
plt.title('Errornorm with Exact Solution of One-shot Approach vs Step-by-Step Approach (time)')
plt.xlabel('time resolution of solve at once')
plt.ylabel('time resolution of solve by step')

for i, (x, y) in enumerate(zip(error_list_full_t, error_list_step_t)):
    plt.text(x, y, str(t_list[i]))

plt.show()


