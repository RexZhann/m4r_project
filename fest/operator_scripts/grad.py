import firedrake as fd

class SpaceTimeOperator():

    def __init__(self,dim, spatial_mesh, layers, layer_height):
        self.dim = dim
        # the space-time mesh of the object
        self.mesh = fd.ExtrudedMesh(spatial_mesh, layers=layers, layer_height=layer_height)

    
    def grad(self, expr):
        return fd.as_vector([expr.dx(i) for i in range(self.dim - 1)])

    # to do: div, curl(dimenstion specific), dt

    def div(self, expr):
        return fd.as_vector(sum([expr[i].dx(i) for i in range(self.dim - 1)]))
    
    def curl(self, expr):
        if self.dim < 4:
            raise ValueError('Curl not defined in this dimension')
        elif self.dim == 4:
            return fd.as_vector([expr[j].dx(i) - expr[i].dx(j) for i in range(self.dim -1) for j in range(self.dim - 1)])
        else:
            pass

    def dt(self, expr):
        return expr.dx(self.dim)
    
    def spatial_coordinate(self):
        ''''''
        return fd.SpatialCoordinate(self.mesh)[:self.dim - 1]
    
    def function_space(self, space_element, time_element, restrict=False):
        if restrict:
            return fd.RestrictedFunctionSpace(fd.FunctionSpace(self.mesh, fd.TemsorProductElement(space_element, time_element)), boundary_set=['bottom'])
        else:
            return fd.FunctionSpace(self.mesh, fd.TemsorProductElement(space_element, time_element))
    
    def initial_condition(function_space, value):
        '''
        Produce a DirichletBC for the initial condition
        param value: expr, function(solution transfer)
        '''




