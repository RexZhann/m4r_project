import firedrake as fd
from .utils import transfer

class SpaceTimeOperator():

    def __init__(self,dim, spatial_mesh, time_steps, step_size, deg=1, cell_shape='interval'):
        '''
        Initialize the spacetime operator object
        '''
        self.dim = dim
        # the space-time mesh of the object
        self.spatial_mesh = spatial_mesh
        self.time_resolution = time_steps
        self.time_terminal = time_steps * step_size
        self.deg = deg
        self.mesh = fd.ExtrudedMesh(spatial_mesh, layers=time_steps, layer_height=time_steps * step_size)
        # layer = time steps, layerheight = time steps * dt

        # no degree, constructing elements
        self.spatial_element_solv = fd.FiniteElement("CG", cell_shape, deg)
        self.temporal_element_solv = fd.FiniteElement("CG", cell_shape, deg)
        self.spatial_element_test = fd.FiniteElement("CG", cell_shape, deg)
        self.temporal_element_test = fd.FiniteElement("DG", cell_shape, deg - 1)
        # passes 


    
    def grad(self, expr):
        '''
        return the spatial gradient vector of the expression
        '''
        return fd.as_vector([expr.dx(i) for i in range(self.dim - 1)])

    # to do: div, curl(dimenstion specific), dt

    def div(self, expr):
        '''
        return the spatial divergence vector of the expression
        '''
        # not a vector, check expr.shape, if more than a vector(tensor)
        return fd.as_vector(sum([expr[i].dx(i) for i in range(self.dim - 1)]))
    
    def curl(self, expr):
        '''
        return the spatial curl vector for 3 dimensional (spatial) expression
        '''

        if self.dim < 4:
            raise ValueError('Curl not defined in this dimension')
        elif self.dim == 4:
            # loop over i, (i+1) % 3
            return fd.as_vector([expr[j].dx(i) - expr[i].dx(j) for i in range(self.dim -1) for j in range(self.dim - 1)])
        else:
            pass

    def dt(self, expr):
        '''
        Return the time derivative
        '''
        return expr.dx(self.dim)
    
    def spatial_coordinate(self):
        '''
        Return the spatial coordinates
        '''
        # issue: Only full slices (:) allowed.
        return tuple(fd.SpatialCoordinate(self.mesh))[:self.dim - 1]
    
    def function_space(self, space_element, time_element, restrict=False):

        if restrict:
            return fd.RestrictedFunctionSpace(fd.FunctionSpace(self.mesh, fd.TensorProductElement(space_element, time_element)), boundary_set=['bottom'])
        else:
            return fd.FunctionSpace(self.mesh, fd.TensorProductElement(space_element, time_element))
    

    def initial_condition(self, function_space, spatial_element, value):
        '''
        Produce a DirichletBC for the initial condition
        param value: expr, function(solution transfer)
        '''
        '''if isinstance(value, fd.Function):
            u_t = transfer(function_space, spatial_element, value, 'top')
            u_b = transfer(function_space, spatial_element, u_t, 'bot')

            return fd.DirichletBC(function_space, u_b)
        else:'''
        return fd.DirichletBC(function_space, value, "bottom")
    
    def get_errornorm(self, h, L, bc_expr, exact_expr, num_iter):
        '''Comparethe difference between numeric method and exact solution'''
        U = self.function_space(self.spatial_element_solv,
                                self.temporal_element_solv)
        V = self.function_space(self.spatial_element_test,
                                self.temporal_element_test)
        U_res = self.function_space(self.spatial_element_solv,
                                    self.temporal_element_solv,
                                    restrict=True)
        u = fd.TrialFunction(U)
        v = fd.TestFunction(V)
        
        x, = fd.SpatialCoordinate(self.spatial_mesh) # Cannot slice
        x_ex, t = fd.SpatialCoordinate(self.mesh)

        sol = fd.Function(U)
        u_2d = fd.Function(U_res)
        u_2d.interpolate(bc_expr(x_ex))
        bc = fd.DirichletBC(U_res, u_2d, 'bottom')

        problem = fd.LinearVariationalProblem(h(u, v), 
                                              L(u,v), 
                                              sol, 
                                              bcs=[bc], 
                                              restrict=True, 
                                              constant_jacobian=True)
        solver = fd.LinearVariationalSolver(problem)

        for _ in range(num_iter):

            solver.solve()
            u_t = transfer(self.spatial_mesh, 
                           self.spatial_element_solv, 
                           sol, 
                           'top', 
                           layer_height=self.time_terminal/num_iter)
            u_b = transfer(self.spatial_mesh, 
                           self.spatial_element_solv, 
                           u_t, 
                           'bottom')

            u_2d.interpolate(u_b, allow_missing_dofs=True)

        err = fd.errornorm(exact_expr(x, self.time_terminal), u_t)

        return err
    
    def refine_time(self, value=2):
        self.time_resolution *= 2
        self.mesh = fd.ExtrudedMesh(self.spatial_mesh, 
                                    layers=self.time_resolution, 
                                    layer_height=self.time_terminal)




