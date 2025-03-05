from fest import SpaceTimeOperator
from firedrake import *
import pytest


spatial_resolution = 100

time_resolution = 100
time_terminal = 10


# Construct spatial mesh
spatial_mesh = UnitIntervalMesh(spatial_resolution)


def test_construction(spatial_mesh=spatial_mesh, time_resolution = time_resolution, time_terminal = time_terminal):

    fest_op = SpaceTimeOperator(dim=2,
                            spatial_mesh=spatial_mesh,
                            time_resolution=time_resolution,
                            time_terminal=time_terminal,
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

    x, t = SpatialCoordinate(fest_op.mesh)

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


    assert errornorm(cos(2*pi*x)*exp(-4*(pi**2)*mu*time_terminal), sol) < 1e-2
    

@pytest.mark.parametrize("deg, time_res, time_term", [(2, 10, 1.0)])
def test_get_errornorm(deg, time_res, time_term):
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
                                 time_resolution=time_res, 
                                 time_terminal=time_term, 
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
    assert error < 1e-5, f"Numerical error is too high: {error}"