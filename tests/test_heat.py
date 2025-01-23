from firedrake import *
from firedrake.pyplot import tripcolor, plot
import matplotlib.pyplot as plt




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
# U_lgmap = U_res.topological.local_to_global_map(None)


V_res = RestrictedFunctionSpace(V, boundary_set=['bottom']) # this changes nothing since DG elelments don't have node on boundary.
# V_lgmap = V_res.topological.local_to_global_map(None)



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

tripcolor(sol1)
plt.show()