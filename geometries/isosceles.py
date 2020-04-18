from dolfin import *
import mshr
import sys
import numpy as np
from capture_cpp_cout import capture_cpp_cout
import matplotlib.pyplot as plt
from fem import *

# Isosceles Triangles Coordinates: (with mass center at 0)
# (-R,-x1), (R,-x1), (0,x2)
#
# -2*x1+x2=0 ## Centroid condition
# (x2+x1)/R=sqrt(3)*AR ## AR=1 means equilateral triangle
# => x2 = 2*x1 => x1 = sqrt(3)*R*AR/3 , x2 = 2*sqrt(3)*R*AR/3
#
# (-R,-sqrt(3)*R*AR/3), (R,-sqrt(3)*R*AR/3), (0,2*sqrt(3)*R*AR/3)

def isosceles(x, R, AR):
    return (x[1]>-sqrt(3)*R*AR/3) and \
           ( x[0]*np.sqrt(3)*AR+2*np.sqrt(3)*R*AR/3>x[1]) and \
           (-x[0]*np.sqrt(3)*AR+2*np.sqrt(3)*R*AR/3>x[1])

def _left_corner(x, a, r):
    return np.linalg.norm(x-np.array( [-1,-np.sqrt(3)*a/3]))<r
def _right_corner(x, a, r):
    return np.linalg.norm(x-np.array( [ 1,-np.sqrt(3)*a/3]))<r
def _upper_corner(x, a, r):
    return np.linalg.norm(x-np.array( [0,2*np.sqrt(3)*a/3]))<r

def corner_isosceles(x, a, r):
    return _upper_corner(x, a, r) or \
           _right_corner(x, a, r) or \
           _left_corner (x, a, r)
def active_isosceles (x, AR): return isosceles(x,1, AR)
def inner_noslip_isosceles (x, R, AR): return isosceles(x,R,AR)
def cross_isosceles (x, width, AR):
    raise NotImplementedError()

def get_isosceles_mesh(_config, res_iterations):
    L = _config['L']

    mesh = RectangleMesh(Point(-L,-L),Point(L,L), _config['res'], _config['res'], 'crossed')
    for i in range(res_iterations):
        cell_markers = MeshFunction("bool", mesh, 2)
        cell_markers.set_all(False)
        for cell in cells(mesh):
            mp = cell.midpoint().array()
            if isosceles(mp, 1.2, _config['AR']):
                cell_markers[cell]=True
        mesh = refine(mesh, cell_markers)
    return mesh

def get_isosceles_bcs(_config, W):
    bcs = []
    # Diagonal BC
    if _config['diagonal_bc']:
        raise NotImplementedError()
    # No-slip center
    if _config['no_slip_center_size']>0:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, lambda x, on_boundary: inner_noslip_isosceles(x, _config['no_slip_center_size'], _config['AR'])))
    return bcs


def get_isosceles_load_vector(_config):
    # Define variational problem
    f = Constant((0.0, 0.0))
    # (-R,-sqrt(3)*R*AR/3), (R,-sqrt(3)*R*AR/3), (0,2*sqrt(3)*R*AR/3)
    if _config['corner_ff']:
        inward_vector = Expression(('-x[0]', '-x[1]'), degree = 2)
        force_norm = Expression(('sqrt(x[0]*x[0]+x[1]*x[1])'), degree = 2)
        domain_upper_corner = Expression('(pow(x[0]-0,2) + pow(x[1]-2*sqrt(3.)*AR/3.,2)<pow(w,2))', degree = 2, w = _config['bar_width'], AR = _config['AR'])
        domain_left_corner  = Expression('(pow(x[0]-1,2) + pow(x[1]+  sqrt(3.)*AR/3.,2)<pow(w,2))', degree = 2, w = _config['bar_width'], AR = _config['AR'])
        domain_right_corner = Expression('(pow(x[0]+1,2) + pow(x[1]+  sqrt(3.)*AR/3.,2)<pow(w,2))', degree = 2, w = _config['bar_width'], AR = _config['AR'])

        f = inward_vector * (domain_upper_corner+ domain_left_corner+domain_right_corner) * Constant(_config['Fscale'])
        if _config['force_scaling']=='const':
            f = f/force_norm
        elif _config['force_scaling']=='inverse':
            f = f/(force_norm*force_norm)
        elif _config['force_scaling']=='proportional':
            pass
        else:
            raise RuntimeError(f"Unrecognized force_scaling: _config['force_scaling']")
    return f

def assemble_isosceles_system(_config):
    res_iterations = _config['initial_res_iterations']
    while True: ## Check if sufficient resolution
        mesh = get_isosceles_mesh(_config, res_iterations)

        W = get_function_space(_config, mesh)
        bcs = get_general_bcs(_config, W) + get_isosceles_bcs(_config, W)
        f = get_isosceles_load_vector(_config)

        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)
        mu = Constant(_config['mu'])
        a = mu*inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
        l = inner(f, v)*dx

        # Form for use in constructing preconditioner matrix
        b = inner(grad(u), grad(v))*dx + p*q*dx

        # Assemble system
        (A, bb),   outstring1 = capture_cpp_cout(lambda : assemble_system(a, l, bcs))
        (P, btmp), outstring2 = capture_cpp_cout(lambda : assemble_system(b, l, bcs))
        if len(outstring1) > 0 or len(outstring2)>0:
            print('There was an issue in the assemble_system function:')
            outstring = outstring1 if len(outstring1)>=len(outstring2) else outstring2
            del outstring1,outstring2
            if "Warning: Found no facets matching domain for boundary condition." in outstring:
                res_iterations += 1
                max_res_iterations = _config['max_res_iterations']
                if res_iterations>max_res_iterations:
                    raise RuntimeError(f"There is probably something wrong, calling assemble_isosceles_system with a res_itertions={res_iterations} larger than {max_res_iterations}... Exiting")
                print(f"Boundary conditions cannot be implemented. Retrying with a higher adaptivity")
                continue
            else:
                print('There was an error in assemble_system. The output of FeniCS:')
                raise RuntimeError(outstring)
        break
    return mesh, W, A, P, bb
