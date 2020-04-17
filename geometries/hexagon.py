from dolfin import *
import mshr
import sys
import numpy as np
from capture_cpp_cout import capture_cpp_cout
import matplotlib.pyplot as plt
from fem import *

# Hexagon Coordinates:
# (-R,0), (-R/2, Sqrt(3)/2 R), ( R/2, Sqrt(3)/2 R),
# ( R,0), ( R/2,-Sqrt(3)/2 R), (-R/2,-Sqrt(3)/2 R)

def hexagon(x, R):
    return (x[1]>-np.sqrt(3)/2*R) and \
           (x[1]< np.sqrt(3)/2*R) and \
           (( np.sqrt(3)*x[0]+x[1] < R*np.sqrt(3))) and \
           (( np.sqrt(3)*x[0]+x[1] >-R*np.sqrt(3))) and \
           ((-np.sqrt(3)*x[0]+x[1] < R*np.sqrt(3))) and \
           ((-np.sqrt(3)*x[0]+x[1] >-R*np.sqrt(3)))
def _corner_hexagon1(x, width, a): return ((x[0]-np.cos(0*np.pi/3+a))**2+(x[1]-np.sin(0*np.pi/3+a))**2<width**2)
def _corner_hexagon2(x, width, a): return ((x[0]-np.cos(1*np.pi/3+a))**2+(x[1]-np.sin(1*np.pi/3+a))**2<width**2)
def _corner_hexagon3(x, width, a): return ((x[0]-np.cos(2*np.pi/3+a))**2+(x[1]-np.sin(2*np.pi/3+a))**2<width**2)
def _corner_hexagon4(x, width, a): return ((x[0]-np.cos(3*np.pi/3+a))**2+(x[1]-np.sin(3*np.pi/3+a))**2<width**2)
def _corner_hexagon5(x, width, a): return ((x[0]-np.cos(4*np.pi/3+a))**2+(x[1]-np.sin(4*np.pi/3+a))**2<width**2)
def _corner_hexagon6(x, width, a): return ((x[0]-np.cos(5*np.pi/3+a))**2+(x[1]-np.sin(5*np.pi/3+a))**2<width**2)

def corner_hexagon(x, width, a):
    return _corner_hexagon1(x, width, a) or \
           _corner_hexagon2(x, width, a) or \
           _corner_hexagon3(x, width, a) or \
           _corner_hexagon4(x, width, a) or \
           _corner_hexagon5(x, width, a) or \
           _corner_hexagon6(x, width, a)
def active_hexagon (x): return hexagon(x,1)
def inner_noslip_hexagon(x, R): return hexagon(x, R)
def cross_hexagon (x, width):
    in_active_area = active_hexagon(x)
    on_one_diagonal = (abs(x[1]) < width) or \
                      (abs(np.sqrt(3)*x[0] - x[1]) < width) or \
                      (abs(np.sqrt(3)*x[0] + x[1]) < width)
    ret = in_active_area and on_one_diagonal and np.linalg.norm(x)>.6
    return ret

def get_hexagon_mesh(_config, res_iterations):
    L = _config['L']

    mesh = RectangleMesh(Point(-L,-L),Point(L,L), _config['res'], _config['res'], 'crossed')
    for i in range(res_iterations):
        cell_markers = MeshFunction("bool", mesh, 2)
        cell_markers.set_all(False)
        for cell in cells(mesh):
            mp = cell.midpoint().array()
            if hexagon(mp, 1.2):
                cell_markers[cell]=True
        mesh = refine(mesh, cell_markers)
    return mesh

def get_hexagon_bcs(_config, W):
    bcs = []
    # Diagonal BC
    if _config['diagonal_bc']:
        velocity_to_center = Expression(("-x[0]*v", "-x[1]*v"), v = Constant(_config['v_scale']), degree=2)
        bcs.append(DirichletBC(W.sub(0), velocity_to_center, lambda x, on_boundary: cross_hexagon(x, _config['bar_width'])))
    # No-slip center
    if _config['no_slip_center_size']>0:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, lambda x, on_boundary: inner_noslip_hexagon(x, _config['no_slip_center_size'])))
    return bcs


def get_hexagonal_load_vector(_config):
    # Define variational problem
    f = Constant((0.0, 0.0))
    if _config['corner_ff']:
        inward_vector = Expression(('-x[0]', '-x[1]'), degree = 2)
        domain = Expression('     (pow(x[0]-cos(0*pi/3+alpha),2) + pow(x[1]-sin(0*pi/3+alpha),2)<pow(w,2))\
                               or (pow(x[0]-cos(1*pi/3+alpha),2) + pow(x[1]-sin(1*pi/3+alpha),2)<pow(w,2))\
                               or (pow(x[0]-cos(2*pi/3+alpha),2) + pow(x[1]-sin(2*pi/3+alpha),2)<pow(w,2))\
                               or (pow(x[0]-cos(3*pi/3+alpha),2) + pow(x[1]-sin(3*pi/3+alpha),2)<pow(w,2))\
                               or (pow(x[0]-cos(4*pi/3+alpha),2) + pow(x[1]-sin(4*pi/3+alpha),2)<pow(w,2))\
                               or (pow(x[0]-cos(5*pi/3+alpha),2) + pow(x[1]-sin(5*pi/3+alpha),2)<pow(w,2))',
                            degree = 1, w = _config['bar_width'], alpha = _config['hexagon_rotation'])
        f = inward_vector * domain * Constant(_config['Fscale'])
    return f

def assemble_hexagon_system(_config):
    res_iterations = _config['initial_res_iterations']
    while True: ## Check if sufficient resolution
        assert(_config['hexagon_rotation']==0.)
        mesh = get_hexagon_mesh(_config, res_iterations)

        W = get_function_space(_config, mesh)
        bcs = get_general_bcs(_config, W) + get_hexagon_bcs(_config, W)
        f = get_hexagonal_load_vector(_config)

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
                    raise RuntimeError(f"There is probably something wrong, calling assemble_hexagon_system with a res_itertions={res_iterations} larger than {max_res_iterations}... Exiting")
                print(f"Boundary conditions cannot be implemented. Retrying with a higher adaptivity")
                continue
            else:
                print('There was an error in assemble_system. The output of FeniCS:')
                raise RuntimeError(outstring)
        break
    return mesh, W, A, P, bb
