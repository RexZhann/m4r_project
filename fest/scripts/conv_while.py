import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from fest import refine_spacetime_alternately, refine_timespace_alternately

mu = 0.000001

def bc_expr(x):
    return cos(2*pi*x)

def h(u, v, mu=mu):
    return (u.dx(1) * v + u.dx(0) * mu * v.dx(0)) * dx

def L(u, v):
    return Constant(0.0) * v * dx

def exact(x, t):
    return cos(2*pi*x)*exp(-4*(pi**2) * mu * t)

deg = 2


sp_hist, t_hist, err_hist = refine_spacetime_alternately(
    sp_init=10,
    t_init=10,
    t_end=1,
    deg=deg,
    bc_expr=bc_expr,
    h=h,
    L=L,
    exact=exact,
    num_iter_mode='step',  
    outer_tol=1e-10,
    max_iter=4,
    return_inner=False
)


fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.loglog([1/n for n in t_hist], err_hist, marker='o')
ax.set_xlabel("dt")
ax.set_ylabel("Error (log scale)")
ax.set_title(f"Alternating refinement of space/time until error < 1e-12(deg {deg})")
ax.grid(True, which='both', linestyle='--', alpha=0.5)


plt.tight_layout()
plt.show()

