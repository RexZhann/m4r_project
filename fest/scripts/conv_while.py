import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from fest import refine_spacetime_alternately, refine_timespace_alternately

mu = 0.0001

def bc_expr(x):
    return cos(2*pi*x)

def h(u, v, mu=mu):
    return (u.dx(1) * v + u.dx(0) * mu * v.dx(0)) * dx

def L(u, v):
    return Constant(0.0) * v * dx

def exact(x, t):
    return cos(2*pi*x)*exp(-4*(pi**2) * mu * t)

deg = 1

sp_hist, t_hist, err_hist, = refine_timespace_alternately(
    sp_init=10,
    t_init=10,
    t_end=0.5,
    deg=deg,
    bc_expr=bc_expr,
    h=h,
    L=L,
    exact=exact,
    num_iter_mode='step',  
    outer_tol=1e-10,
    max_iter=8,
    return_inner=False
)


fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.loglog([1/n for n in sp_hist], err_hist, marker='o') # use different markers for different lines (not just color)
ax.set_xlabel("dt")
ax.set_ylabel("Error ")
ax.set_title(f"Convergence Rate of Errors in Time d until error < 1e-12 (degree {deg})") # display degrees on legend, put title info in captions
# put space and time plots next to each other (text size same as report text size)
ax.grid(True, which='both', linestyle='--', alpha=0.5)


plt.tight_layout()
plt.show()

