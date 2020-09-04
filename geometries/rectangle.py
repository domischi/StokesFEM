from dolfin import *
import mshr
import sys
from capture_cpp_cout import capture_cpp_cout
import matplotlib.pyplot as plt
from fem import *

def active_rect (x, on_boundary, AR, bar_width): return x[0]>-1 and x[0]< 1 and x[1]>-AR and x[1]<AR
def diagonal_up (x, on_boundary, AR, bar_width): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]-AR) < bar_width ## rising diagonal
def diagonal_dw (x, on_boundary, AR, bar_width): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]+AR) < bar_width ## lowering diagonal
def cross_rect  (x, on_boundary, AR, bar_width): return active_rect(x, on_boundary, AR, bar_width) and (diagonal_up(x, on_boundary, AR, bar_width) or diagonal_dw(x, on_boundary, AR, bar_width))
def corner      (x, on_boundary, AR, bar_width, R=1): return abs(abs(x[0])-R)<bar_width and abs(abs(x[0])-R*AR)<bar_width
def inner_noslip_rectangular(x, on_boundary, AR, R): return x[0]>-R and x[0]< R and x[1]>-AR*R and x[1]<AR*R

def get_rectangular_mesh(_config, res_iterations):
    AR = _config['AR']
    L = _config['L']
    mesh = RectangleMesh(Point(-L,-L),Point(L,L), _config['res'], _config['res'], 'left/right')
    for i in range(res_iterations):
        cell_markers = MeshFunction("bool", mesh, 2)
        cell_markers.set_all(False)
        for cell in cells(mesh):
            mp = cell.midpoint().array()
            if mp[0]>-1.2 and mp[0]<1.2 and  mp[1]>-1.2*AR and mp[1]<1.2*AR:
                cell_markers[cell]=True
        mesh = refine(mesh, cell_markers)
    return mesh

def get_rectangular_bcs(_config, W):
    AR = _config['AR']
    bcs = []
    # Diagonal BC
    if _config['diagonal_bc']:
        velocity_to_center = Expression(("-x[0]*v", "-x[1]*v"), v = Constant(_config['v_scale']), degree=2)
        bcs.append(DirichletBC(W.sub(0), velocity_to_center, lambda x, on_boundary: cross_rect(x, on_boundary, AR, _config['bar_width'])))
    # No-slip center
    if _config['no_slip_center_size']>0:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, lambda x, on_boundary: inner_noslip_rectangular(x, on_boundary, AR, _config['no_slip_center_size'])))
    return bcs


def get_load_vector(_config):
    # Define variational problem
    f = Constant((0.0, 0.0))
    if _config['rectangular_ff']:
        inward_vector = Expression(('-x[0]', '-x[1]'), degree = 2)
        domain = Expression('x[0]>-1 and x[0]< 1 and x[1]>-AR and x[1]<AR', degree = 1, AR = _config['AR'])
        f = inward_vector * domain* Constant(_config['Fscale'])
    elif _config['diagonal_ff']:
        inward_vector = Expression(('-x[0]', '-x[1]'), degree = 2)
        domain = Expression('(abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]-AR) < bar_width) or (abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]+AR) < bar_width)', degree = 1, AR = _config['AR'], bar_width = _config['bar_width'])
        f = inward_vector * domain * Constant(_config['Fscale'])
    elif _config['corner_ff']:
        inward_vector = Expression(('-x[0]', '-x[1]'), degree = 2)
        domain = Expression('(abs(abs(x[0])-R)<bar_width and abs(abs(x[1])-R*AR) < bar_width)', degree = 2, AR = _config['AR'], bar_width = _config['bar_width'], R=_config.get("corner_force_distance", 1.))
        f = inward_vector * domain * Constant(_config['Fscale'])
    return f

def assemble_rectangular_system(_config):
    res_iterations = _config['initial_res_iterations']
    while True: ## Check if sufficient resolution
        mesh = get_rectangular_mesh(_config, res_iterations)

        W = get_function_space(_config, mesh)
        bcs = get_general_bcs(_config, W) + get_rectangular_bcs(_config, W)
        f = get_load_vector(_config)

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
                    raise RuntimeError(f"There is probably something wrong, calling assemble_rectangular_system with a res_itertions={res_iterations} larger than {max_res_iterations}... Exiting")
                print(f"Boundary conditions cannot be implemented. Retrying with a higher adaptivity")
                continue
            else:
                print('There was an error in assemble_system. The output of FeniCS:')
                raise RuntimeError(outstring)
        break
    return mesh, W, A, P, bb
