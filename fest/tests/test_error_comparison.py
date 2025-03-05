from fest import SpaceTimeOperator
import firedrake as fd
import pytest


# Part 1: Construct the basics

# Construct spatial mesh
spatial_mesh = fd.UnitIntervalMesh(5)


# Initialize the space-time operator
fest_op = SpaceTimeOperator(dim=2,
                            spatial_mesh=spatial_mesh,
                            layers=10,
                            layer_height=0.5,
                            deg=1
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

V_res = fest_op.function_space(space_element=fest_op.spatial_element_test,
                           time_element=fest_op.temporal_element_test,
                           restrict=True
                           )

u = fd.TrialFunction(U)

v = fd.TestFunction(V)

u_init = fd.Function(U_res)

x, t = fest_op.spatial_coordinate()

bc = fest_op.initial_condition(function_space=U_res,
                               spatial_element=fest_op.spatial_element_solv,
                               value=u_init
                               )

# Part 2: Build up the solve-at-once method and solve-by-step method

