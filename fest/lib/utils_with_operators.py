from fest import SpaceTimeOperator
import firedrake as fd

def transfer(m, space_element, sol1, pos, layer_height=0.5):
    """
    Perform the extract-reinsert algorithm using immersed mesh

    param m: the original 1D mesh (unextruded dimension)
    param space_element: the spatial finite element (FE associated with the original mesh)
    param sol1: the function being extracted / reinserted
    param pos: a string being either 'top' or 'bottom'

    return u_1d: a 1d function being extracted out
    return u_f: a 2D function being re-inserted into the mesh
    """
    # Create the function space for the top of the extrusion
    Fs_imm = fd.VectorFunctionSpace(m, 'CG', 1, dim=2)  
    x_f = fd.Function(Fs_imm)  # Create the storer function in Fs_top

    # Get the spatial coordinate
    x, = fd.SpatialCoordinate(m)

    # Interpolate the storer with information of the pos
    if pos == 'bottom':
        x_f.interpolate(fd.as_vector([x, 0]))
    elif pos == 'top':
        x_f.interpolate(fd.as_vector([x, layer_height]))
    else:
        raise NotImplementedError

    # Create the immersed mesh (1D mesh in 2D space)
    m_imm = fd.Mesh(x_f)
    # Define the function space on the immersed mesh and interpolate the solution
    UFs_imm = fd.FunctionSpace(m_imm, space_element)
    u_f = fd.Function(UFs_imm)
    if pos == 'top':
        u_f.interpolate(sol1, allow_missing_dofs=True)
        # Define the function space on the original mesh
        UFs_1D = fd.FunctionSpace(m, space_element)
        u_1d = fd.Function(UFs_1D)
    
        u_1d.dat.data_wo[:] = u_f.dat.data_ro

        return u_1d
    else:
        u_f.dat.data_wo[:] = sol1.dat.data_ro
        return u_f


def refine_spacetime(operator, mode='step'):
    err = operator.get_errornorm()