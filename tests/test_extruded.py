from firedrake import *

#
#                  rank 0                 rank 1
#
#  plex points:
#
#            +-------+-------+      +-------+-------+
#            |       |       |      |       |       |
#            |       |       |      |       |       |
#            |       |       |      |       |       |
#            +-------+-------+      +-------+-------+
#            2   0  (3) (1) (4)    (4) (1)  2   0   3    () = ghost
#
#  mesh._dm_renumbering:
#
#            [0, 2, 3, 1, 4]        [0, 3, 2, 1, 4]
#
#  Local DoFs:
#
#            5---2--(8)(11)(14)   (14)(11)--8---2---5
#            |       |       |      |       |       |
#            4   1  (7)(10)(13)   (13)(10)  7   1   4
#            |       |       |      |       |       |
#            3---0--(6)-(9)(12)   (12)-(9)--6---0---3    () = ghost
#
#  Global DoFs:
#
#                       3---1---9---5---7
#                       |       |       |
#                       2   0   8   4   6
#                       |       |       |
#                       x---x---x---x---x
#
#  LGMap:
#
#    rank 0 : [-1, 0, 1, -1, 2, 3, -1, 8, 9, -1, 4, 5, -1, 6, 7]
#    rank 1 : [-1, 4, 5, -1, 6, 7, -1, 8, 9, -1, 0, 1, -1, 2, 3]
mesh = UnitIntervalMesh(2)
extm = ExtrudedMesh(mesh, 1)
V = FunctionSpace(extm, "CG", 2)
V_res = RestrictedFunctionSpace(V, boundary_set=["bottom"])
# Check lgmap.
lgmap = V_res.topological.local_to_global_map(None)
if mesh.comm.rank == 0:
    lgmap_expected = [-1, 0, 1, -1, 2, 3, -1, 8, 9, -1, 4, 5, -1, 6, 7]
else:
    lgmap_expected = [-1, 4, 5, -1, 6, 7, -1, 8, 9, -1, 0, 1, -1, 2, 3]

print(lgmap.indices)
# assert np.allclose(lgmap.indices, lgmap_expected)

# Check vec.
n = V_res.dof_dset.size
lgmap_owned = lgmap.indices[:n]
local_global_filter = lgmap_owned >= 0
local_array = 1.0 * np.arange(V_res.dof_dset.total_size)
f = Function(V_res)
f.dat.data_wo_with_halos[:] = local_array
with f.dat.vec as v:
    assert np.allclose(v.getArray(), local_array[:n][local_global_filter])
    v *= 2.
assert np.allclose(f.dat.data_ro_with_halos[:n][local_global_filter], 2. * local_array[:n][local_global_filter])

# Solve Poisson problem.
x, y = SpatialCoordinate(extm)
normal = FacetNormal(extm)
exact = Function(V_res).interpolate(x**2 * y**2)
exact_grad = as_vector([2 * x * y**2, 2 * x**2 * y])
u = TrialFunction(V_res)
v = TestFunction(V_res)
a = inner(grad(u), grad(v)) * dx
L = inner(-2 * (x**2 + y**2), v) * dx + inner(dot(exact_grad, normal), v) * ds_v(2) + inner(dot(exact_grad, normal), v) * ds_t
bc = DirichletBC(V_res, exact, "bottom")
sol = Function(V_res)
solve(a == L, sol, bcs=[bc])
assert assemble(inner(sol - exact, sol - exact) * dx)**0.5 < 1.e-15