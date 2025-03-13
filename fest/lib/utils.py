from firedrake import *
from firedrake.pyplot import tripcolor, plot 



def transfer(m, W_s, sol1, pos, layer_height=0.5):
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

    u = TrialFunction(U)
    v = TestFunction(V)


    x, = SpatialCoordinate(m)
    x_ex, t = SpatialCoordinate(mesh_ex)

    

    sol_0 = Function(U)
    u_2d = Function(U_res)
    u_2d.interpolate(bc_expr(x_ex))
    bc = DirichletBC(U_res, u_2d, 'bottom')

    sol = sol_0
    problem = LinearVariationalProblem(h(u, v), L(u,v), sol, bcs=[bc], restrict=True, constant_jacobian=True)
    solver = LinearVariationalSolver(problem)

    for _ in range(num_iter):

        solver.solve()
        u_t = transfer(m, W_s, sol, 'top', layer_height=t_end/num_iter)
        u_b = transfer(m, W_s, u_t, 'bottom')

        u_2d.interpolate(u_b, allow_missing_dofs=True)
    
    err = errornorm(exact(x, t_end), u_t)

    return err

def refine_spacetime_alternately(sp_init=10, t_init=10, t_end=0.5, deg=1,
                                 bc_expr=None, h=None, L=None, exact=None,
                                 num_iter_mode='step',  # or 'once'
                                 inner_tol=0.1,       
                                 outer_tol=1e-12, 
                                 max_iter=30,
                                 max_inner_iter=5,
                                 return_inner=True):
    """
    Switch between refining spatial resolution and time resolution,

    param:
    - spatial_init: Initial number of cells on spatial dimensions
    - time_init: Initial number of cells on time dimension
    - t_end:   Terminal time
    - deg:     degree of polynomial of finite elements
    - bc_expr, h, L, exact: UFL expressions used for boundary conditions,
                            equation to be solved, and exact solution
    - num_iter_mode:
       * 'once': solve at once, num_iter=1
       * 'step': solve by steps, num_iter = t_res
    - inner_tol: the tolerance of amount of change of error when refining spatial resolution
    - tol: tolerance of error
    - max_iter: maximum number of iterations
    - return_inner: if true, will append the err hist from inner loops

    return:
    - sp_history: iteration history of spatial resolution
    - t_history:  iteration history of time resolution
    - err_history: iteration history of error 
    """

    # Initialization
    sp_res = sp_init
    t_res = t_init
    err = get_spacetime_errornorm(
        sp_res=sp_res,
        t_res=t_res,
        t_end=t_end,
        deg=deg,
        bc_expr=bc_expr,
        h=h,
        L=L,
        exact=exact,
        num_iter=(t_res if num_iter_mode == 'step' else 1)
    )


    sp_history = [sp_res]
    t_history = [t_res]
    err_history = [err]

    iteration = 0

    # decide using 'solve at once' or 'solve by step'
    def get_num_iter(t_r, mode):
        return (t_r if mode == 'step' else 1)

    # outer loop
    while err > outer_tol and iteration < max_iter:
        iteration += 1
        print(f"\n--- Outer iteration {iteration}, err={err:.3e} ---")

        # ------------ (A) keep refining space------------
        inner_iter = 0
        while True:
            inner_iter += 1
            old_err = err  

            sp_res *= 2   # space refining strategy

            # compute new error
            err = get_spacetime_errornorm(
                sp_res=sp_res,
                t_res=t_res,
                t_end=t_end,
                deg=deg,
                bc_expr=bc_expr,
                h=h,
                L=L,
                exact=exact,
                num_iter=get_num_iter(t_res, num_iter_mode)
            )
            rel_diff = abs(err - old_err)/abs(old_err)

            print(f"   Refine space: sp_res={sp_res}, old_err={old_err:.3e}, new_err={err:.3e}, rel_diff={rel_diff:.3f}")

            if return_inner:
                sp_history.append(sp_res)
                t_history.append(t_res)
                err_history.append(err)

            # Compare with threshold1
            if rel_diff < inner_tol or (inner_iter) >= max_inner_iter:
                break
        
        sp_res /= 2
        # ------------ (B) refine time ------------
        t_res *= 2
        old_err = err

        err = get_spacetime_errornorm(
            sp_res=sp_res,
            t_res=t_res,
            t_end=t_end,
            deg=deg,
            bc_expr=bc_expr,
            h=h,
            L=L,
            exact=exact,
            num_iter=get_num_iter(t_res, num_iter_mode)
        )

        sp_history.append(sp_res)
        t_history.append(t_res)
        err_history.append(err)
        print(f"   Refine time:  t_res={t_res}, old_err={old_err:.3e}, new_err={err:.3e}")


        print(f"End of outer iteration {iteration}, final err={err:.3e}")
    # END while

    print(f"\nRefinement finished after {iteration} outer iterations.")
    print(f"Final: sp_res={sp_res}, t_res={t_res}, err={err:.6g}")
    return sp_history, t_history, err_history


def refine_timespace_alternately(sp_init=10, t_init=10, t_end=0.5, deg=1,
                                 bc_expr=None, h=None, L=None, exact=None,
                                 num_iter_mode='step',  # or 'once'
                                 inner_tol=0.001,       
                                 outer_tol=1e-12, 
                                 max_iter=30,
                                 max_inner_iter=5,
                                 return_inner=True):
    """
    Switch between refining spatial resolution and time resolution,

    param:
    - spatial_init: Initial number of cells on spatial dimensions
    - time_init: Initial number of cells on time dimension
    - t_end:   Terminal time
    - deg:     degree of polynomial of finite elements
    - bc_expr, h, L, exact: UFL expressions used for boundary conditions,
                            equation to be solved, and exact solution
    - num_iter_mode:
       * 'once': solve at once, num_iter=1
       * 'step': solve by steps, num_iter = t_res
    - inner_tol: the tolerance of amount of change of error when refining spatial resolution
    - tol: tolerance of error
    - max_iter: maximum number of iterations
    - return_inner: if true, will append the err hist from inner loops

    return:
    - sp_history: iteration history of spatial resolution
    - t_history:  iteration history of time resolution
    - err_history: iteration history of error 
    """

    # Initialization
    sp_res = sp_init
    t_res = t_init
    err = get_spacetime_errornorm(
        sp_res=sp_res,
        t_res=t_res,
        t_end=t_end,
        deg=deg,
        bc_expr=bc_expr,
        h=h,
        L=L,
        exact=exact,
        num_iter=(t_res if num_iter_mode == 'step' else 1)
    )


    sp_history = [sp_res]
    t_history = [t_res]
    err_history = [err]

    iteration = 0

    # decide using 'solve at once' or 'solve by step'
    def get_num_iter(t_r, mode):
        return (t_r if mode == 'step' else 1)

    # outer loop
    while err > outer_tol and iteration < max_iter:
        iteration += 1
        print(f"\n--- Outer iteration {iteration}, err={err:.3e} ---")

        # ------------ (A) keep refining space------------
        inner_iter = 0
        while True:
            inner_iter += 1
            old_err = err  

            t_res *= 2   # space refining strategy

            # compute new error
            err = get_spacetime_errornorm(
                sp_res=sp_res,
                t_res=t_res,
                t_end=t_end,
                deg=deg,
                bc_expr=bc_expr,
                h=h,
                L=L,
                exact=exact,
                num_iter=get_num_iter(t_res, num_iter_mode)
            )
            rel_diff = abs(err - old_err)/abs(old_err)



            print(f"   Refine space: sp_res={sp_res}, old_err={old_err:.3e}, new_err={err:.3e}, rel_diff={rel_diff:.3f}")

            if return_inner:
                sp_history.append(sp_res)
                t_history.append(t_res)
                err_history.append(err)

            # Compare with threshold1
            if rel_diff < inner_tol or (inner_iter) >= max_inner_iter:
                break
        
        t_res /= 2
        t_res = int(t_res)
        # ------------ (B) refine time ------------
        sp_res *= 2
        old_err = err

        err = get_spacetime_errornorm(
            sp_res=sp_res,
            t_res=t_res,
            t_end=t_end,
            deg=deg,
            bc_expr=bc_expr,
            h=h,
            L=L,
            exact=exact,
            num_iter=get_num_iter(t_res, num_iter_mode)
        )

        sp_history.append(sp_res)
        t_history.append(t_res)
        err_history.append(err)
        conv_order_t = np.log(old_err/err) / np.log(2)
        print(f"   Refine time:  t_res={t_res}, old_err={old_err:.3e}, new_err={err:.3e}, conv_rate={conv_order_t:.3f}")


        print(f"End of outer iteration {iteration}, final err={err:.3e}")
    # END while

    print(f"\nRefinement finished after {iteration} outer iterations.")
    print(f"Final: sp_res={sp_res}, t_res={t_res}, err={err:.6g}")
    return sp_history, t_history, err_history