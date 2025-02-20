import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from fest import get_spacetime_errornorm

mu = 0.001

def bc_expr(x):
    return cos(2*pi*x)

def h(u, v, mu=mu):
    return (u.dx(1) * v + u.dx(0) * mu * v.dx(0)) * dx

def L(u, v):
    return Constant(0.0) * v * dx

def exact(x, t):
    return cos(2*pi*x)*exp(-4*(pi**2) * mu * t)



'''def refine_spacetime_alternately(sp_init=10, t_init=1, t_end=0.5, deg=3, 
                                 bc_expr=None, h=None, L=None, exact=None,
                                 num_iter_mode='step',  # or 'once'
                                 tol=1e-12, max_iter=30):
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
    - tol: tolerance of error
    - max_iter: maximum number of iterations

    return:
    - sp_history: iteration history of spatial resolution
    - t_history:  iteration history of time resolution
    - err_history: iteration history of error 
    """
    sp_res = sp_init
    t_res = t_init
    
    # Strings to store iter history
    sp_history = []
    t_history = []
    err_history = []
    
    iteration = 0

    err = 1e20
    err_0 = get_spacetime_errornorm(
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
    err_history.append(err_0)

    def get_num_iter(t_r, mode):
        if mode == 'step':
            return t_r
        else:  # mode == 'once'
            return 1

    pointer = 0 # indicate which dimension of resolution to refine
    
    
    while err > tol and iteration < max_iter:
        inner_iteration = 0
        while (abs(err - err_history[-1]) / err_history[-1]) > 0.1 and inner_iteration < max_iter:
            print(f'starting iteration{iteration}, inner_iteration{inner_iteration}')
        

            if err <= tol:
                break

            if iteration > max_iter:  
                break
            
            sp_res += 10


            # Calculate the new error
            err = get_spacetime_errornorm(sp_res=sp_res,
                                        t_res=t_res,
                                        t_end=t_end,
                                        deg=deg,
                                        bc_expr=bc_expr,
                                        h=h,
                                        L=L,
                                        exact=exact,
                                        num_iter=get_num_iter(t_res, num_iter_mode))
            inner_iteration += 1

        t_res += 1


        sp_history.append(sp_res)
        t_history.append(t_res)
        err_history.append(err)

        iteration += 1

        print(f"Iter={iteration}, sp_res={sp_res}, t_res={t_res}, err={err}")

    
    print(f"Final sp_res={sp_res}, t_res={t_res}, err={err}, iteration={iteration}")

    return sp_history, t_history, err_history'''

def refine_spacetime_alternately(sp_init=10, t_init=1, t_end=0.5, deg=3,
                                 bc_expr=None, h=None, L=None, exact=None,
                                 num_iter_mode='step',  # or 'once'
                                 inner_tol=0.1,       
                                 outer_tol=1e-12, max_iter=30):
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
        print(f"\n=== Outer iteration {iteration}, err={err:.3e} ===")

        # ------------ (A) keep refining space------------
        inner_iter = 0
        while True:
            inner_iter += 1
            old_err = err  

            sp_res += 10   # space refining strategy

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

            
            sp_history.append(sp_res)
            t_history.append(t_res)
            err_history.append(err)

            # Compare with threshold1
            if rel_diff < inner_tol or (inner_iter + iteration) >= max_iter:
                break

        # ------------ (B) refine time ------------
        t_res += 1
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

deg = 2


sp_hist, t_hist, err_hist = refine_spacetime_alternately(
    sp_init=10,
    t_init=1,
    t_end=0.5,
    deg=deg,
    bc_expr=bc_expr,
    h=h,
    L=L,
    exact=exact,
    num_iter_mode='step',  
    tol=1e-12,
    max_iter=30
)


fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.semilog(t_hist, err_hist, marker='o')
ax.set_xlabel("number of time cells")
ax.set_ylabel("Error (log scale)")
ax.set_title(f"Alternating refinement of space/time until error < 1e-12(deg {deg})")
ax.grid(True, which='both', linestyle='--', alpha=0.5)


plt.tight_layout()
plt.show()

