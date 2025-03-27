import os
import numpy as np
import matplotlib.pyplot as plt
from firedrake import *
from fest import refine_spacetime_alternately, refine_timespace_alternately



mu = 0.0001

def bc_expr(x):
    return cos(2*np.pi*x)

def h(u, v, mu=mu):
    return (u.dx(1) * v + u.dx(0) * mu * v.dx(0)) * dx

def L(u, v):
    return Constant(0.0) * v * dx

def exact(x, t):
    return cos(2*np.pi*x)*exp(-4*(np.pi**2) * mu * t)

# List of polynomial degrees to sweep over
degrees = [1, 2, 3, 4]

markers = ['o', '<', '>', "^"]

# Prepare for plotting
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

results_file = "results.txt"
file_exists = os.path.exists(results_file)

 # Function to draw reference triangle (for convergence rate illustration)
def draw_reference_triangle(ax, base_point, dx_log, slope, label):
    """
    Draws a slope triangle on a log-log plot and labels the horizontal and vertical edges.
    - base_point: (x, y) where the triangle starts (lower-right corner)
    - dx_log: horizontal step in log10 scale (e.g., 1 means one order of magnitude)
    - slope: convergence rate
    - label: slope label (e.g., "1", "2", etc.)
    """
    
    x0, y0 = base_point
    x1 = x0 / (10 ** 0.2)
    y1 = y0 / (10 ** (0.2 * slope))

    # Draw triangle edges
    ax.plot([x0, x0], [y0, y1], 'k-', linewidth=1)  # vertical
    ax.plot([x0, x1], [y1, y1], 'k-', linewidth=1)  # horizontal
    ax.plot([x0, x1], [y0, y1], 'k-', linewidth=1)  # hypotenuse

    # Place labels at midpoints
    ax.text((x0 + x1) / 2, y1 * 0.8, "1", fontsize=9, ha='center', va='bottom')
    ax.text(x0 * 1.2, (y0 + y1) / 2, f"{deg}", fontsize=9, ha='right', va='center')



for deg in degrees:
    # Perform refinements for this degree
    sp_hist, t_hist, err_hist, conv_rate = refine_timespace_alternately(
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
    
    # Append results to file (create if not existing)
    with open(results_file, "a") as f:
        # Write a header only if the file does not exist yet
        if not file_exists:
            f.write("degree,sp_hist,t_hist,err_hist\n")
            file_exists = True  # ensure we don't write the header again
        
        for sp_val, t_val, err_val in zip(sp_hist, t_hist, err_hist):
            f.write(f"{deg},{sp_val},{t_val},{err_val}\n")
    
    # Plot: marker='o' is just an example; you can choose different markers
    ax.loglog([1.0/n for n in sp_hist], err_hist, marker=markers[deg - 1], label=f"Degree {deg}")
    draw_reference_triangle(ax, base_point=(1.2 / sp_hist[-3] , err_hist[-3] * 0.9), dx_log=0.5, slope=conv_rate, label=f"{deg}")



# Add labels and legend to the plot
ax.set_xlabel("dt")
ax.set_ylabel("Error")
# ax.set_title("Convergence Rate of Errors in Time\nuntil error < 1e-12 for various polynomial degrees")
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.show()