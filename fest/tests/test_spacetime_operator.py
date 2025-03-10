from fest import SpaceTimeOperator
from firedrake import *
import pytest


spatial_resolution = 100

time_steps = 100
step_size = 0.1
time_terminal = 10


# Construct spatial mesh
spatial_mesh = UnitIntervalMesh(spatial_resolution)


mesh = ExtrudedMesh(spatial_mesh, layers=time_steps, layer_height=time_steps*step_size)

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


V_res = RestrictedFunctionSpace(V, boundary_set=['bottom']) # this changes nothing since DG elelments don't have node on boundary.


u = TrialFunction(U)

v = TestFunction(V)

u_init = Function(U_res)

x, t = SpatialCoordinate(mesh)

expr_old = (u.dx(1) * v - u * v.dx(0))


def test_mesh(spatial_mesh=spatial_mesh, time_steps=time_steps, step_size=step_size):
    fest_op = SpaceTimeOperator(dim=2,
                            spatial_mesh=spatial_mesh,
                            time_steps=time_steps,
                            step_size=step_size,
                            )
    
    pass



def test_function_space():
    fest_op = SpaceTimeOperator(dim=2,
                            spatial_mesh=spatial_mesh,
                            time_steps=time_steps,
                            step_size=step_size,
                            )
    pass

@pytest.mark.parametrize("sp_deg, t_deg", [(1, 1), (3, 1), (7, 1)])
def test_grad(sp_deg, t_deg, spatial_mesh=spatial_mesh, time_steps=time_steps, step_size=step_size):
    fest_op = SpaceTimeOperator(dim=2,
                            spatial_mesh=spatial_mesh,
                            time_steps=time_steps,
                            step_size=step_size,
                            )
    
    spatial_element_solv = FiniteElement("CG", "interval", sp_deg)
    temporal_element_solv = FiniteElement("CG", "interval", t_deg)

    U_sp = FunctionSpace(mesh, spatial_element_solv)


    U_fest = fest_op.function_space(space_element=spatial_element_solv, 
                               time_element=temporal_element_solv,
                               restrict=False
                               )
    
    u_sp = Function(U_sp)
    u_fest = Function(U_fest)
    

    x_fest, = fest_op.spatial_coordinate()
    x_sp  = SpatialCoordinate(mesh)[0]

    u_fest.interpolate(cos(x_fest))
    u_sp.interpolate(cos(x_sp))
    print(len(u_fest.ufl_shape))
    print(len(u_sp.ufl_shape))

    assert errornorm(fest_op.grad(u_fest), u_sp.dx(0)) < 1e-16

def test_div(spatial_mesh=spatial_mesh, time_steps=time_steps, step_size=step_size):
    pass

def test_curl(spatial_mesh=spatial_mesh, time_steps=time_steps, step_size=step_size):
    pass


def test_construction(spatial_mesh=spatial_mesh, time_steps = time_steps, time_terminal=time_terminal):

    fest_op = SpaceTimeOperator(dim=2,
                            spatial_mesh=spatial_mesh,
                            time_steps=time_steps,
                            step_size=step_size,
                            deg=2
                            )
    # Construct the solve and test function space
    U = fest_op.function_space(space_element=fest_op.spatial_element_solv,
                            time_element=fest_op.temporal_element_solv
                            )

    V = fest_op.function_space(space_element=fest_op.spatial_element_test,
                            time_element=fest_op.temporal_element_test
                            )

    # Construct the solve and test function space
    U_res = fest_op.function_space(space_element=fest_op.spatial_element_solv,
                            time_element=fest_op.temporal_element_solv,
                            restrict=True
                            )

    u = TrialFunction(U)

    v = TestFunction(V)

    u_init = Function(U_res)

    x, = fest_op.spatial_coordinate()

    u_init.interpolate(cos(2*pi*x))

    bc = fest_op.initial_condition(function_space=U_res,
                                spatial_element=fest_op.spatial_element_solv,
                                value=u_init
                                )
    
    sol = Function(U)

    mu = 0.0001
    # LHS for Linear Advection equation
    a = (u.dx(1) * v - u * mu * v.dx(0)) * dx


    # LHS for Heat Equation
    h = (u.dx(1) * v  + u.dx(0) * mu * v.dx(0)) * dx

    L = Constant(0.0) * v * dx

    solve(h == L, sol, bcs=[bc], restrict=True)

    pass
    # assert errornorm(cos(2*pi*x)*exp(-4*(pi**2)*mu*time_terminal), sol) < 1e-2
    

@pytest.mark.parametrize("deg, time_steps, step_size", [(2, 10, 1.0)])
def test_get_errornorm(deg, time_steps, step_size):
    """
    Test the get_errornorm() method of SpaceTimeOperator.
    Ensures that the numerical solution approximates the exact solution well.
    """

    # Create a simple 1D mesh
    nx = 10
    spatial_mesh = UnitIntervalMesh(nx)
    
    # Define SpaceTimeOperator
    operator = SpaceTimeOperator(dim=2, 
                                 spatial_mesh=spatial_mesh, 
                                 time_steps=time_steps, 
                                 step_size=step_size, 
                                 deg=deg)

    mu = 0.00001
    def bc_expr(x):
        return cos(2*pi*x)

    def h(u, v, mu=mu):
        return (u.dx(1) * v + u.dx(0) * mu * v.dx(0)) * dx

    def L(u, v):
        return Constant(0.0) * v * dx

    def exact_expr(x, t):
        return cos(2*pi*x)*exp(-4*(pi**2) * mu * t)


    # Run get_errornorm()
    error = operator.get_errornorm(h, L, bc_expr, exact_expr, num_iter=1)

    # Assert error is small
    assert error < 1e-3, f"Numerical error is too high: {error}"