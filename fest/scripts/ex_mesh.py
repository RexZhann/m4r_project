from firedrake import *
from firedrake.pyplot import tripcolor, plot
import matplotlib.pyplot as plt



m = UnitIntervalMesh(5)
mesh = ExtrudedMesh(m, layers=10, layer_height=2)

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
print("u_init's space:", u_init.function_space())
print("U_res         :", U_res)
print("Is same object?", u_init.function_space() is U_res)


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



Fs_top = VectorFunctionSpace(m, 'CG', 1, dim=2) # This is the function space for the top of the extrusion
x_top = Function(Fs_top) # This is the storer function in Fs_top
x, = SpatialCoordinate(m) # This is the spatial coordinate of x, since the top of extrusion should have same ordering with the original mesh
x_top.interpolate(as_vector([x, 20])) # Interpolate the storer with information of the top

m_top = Mesh(x_top) # the immersed mesh, that is, the 1d mesh immersed in 2d space.
UFs_top = FunctionSpace(m_top, W_s) 
u_top = Function(UFs_top)
u_top.interpolate(sol1)

UFs_1D = FunctionSpace(m, W_s)
u_1d = Function(UFs_1D)
u_1d.dat.data_wo[:] = u_top.dat.data_ro


Fs_bot = VectorFunctionSpace(m, 'CG', 1, dim=2) # This is the function space for the bot of the extrusion
x_bot = Function(Fs_bot) # This is the storer function in Fs_top
x_bot.interpolate(as_vector([x, 0])) # Interpolate the storer with information of the bot

m_bot = Mesh(x_bot) # the immersed mesh, that is, the 1d mesh immersed in 2d space.
UFs_bot = FunctionSpace(m_bot, W_s) 
u_bot = Function(UFs_bot)
u_bot.dat.data_wo[:] = u_1d.dat.data_ro

u_2d = Function(U_res)
u_2d.interpolate(u_bot, allow_missing_dofs=True)


# u_2d.dat.data_wo[:len(u_1d.dat.data_ro)] = u_1d.dat.data_ro

bc_renew = DirichletBC(U_res, u_2d, 'bottom')

sol2 = Function(U_res)

solve(h == L, sol2, bcs=[bc_renew], restrict=True)

plot(u_1d)
plt.show()