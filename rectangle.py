from dolfin import *
import mshr
import sys
from capture_cpp_cout import capture_cpp_cout
import matplotlib.pyplot as plt

def active_rect (x, on_boundary, AR, bar_width): return x[0]>-1 and x[0]< 1 and x[1]>-AR and x[1]<AR
def diagonal_up (x, on_boundary, AR, bar_width): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]-AR) < bar_width ## rising diagonal
def diagonal_dw (x, on_boundary, AR, bar_width): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]+AR) < bar_width ## lowering diagonal
def cross       (x, on_boundary, AR, bar_width): return active_rect(x, on_boundary, AR, bar_width) and (diagonal_up(x, on_boundary, AR, bar_width) or diagonal_dw(x, on_boundary, AR, bar_width))
def left_right  (x, on_boundary, L            ): return x[0] > L * (1-DOLFIN_EPS) or x[0] < L * (-1+DOLFIN_EPS)
def top_bottom  (x, on_boundary, L            ): return x[1] > L * (1-DOLFIN_EPS) or x[1] < L * (-1+DOLFIN_EPS)
def inner_noslip(x, on_boundary, AR, R        ): return x[0]>-R and x[0]< R and x[1]>-AR*R and x[1]<R

def solve_rectangle(_config):
    max_res_iterations = 5
    if _config['res_iterations']>max_res_iterations:
        print(f'There is probably something wrong, calling solve_rectangle with a res_itertions larger than {max_res_iterations}... Exiting')
        sys.exit()
    AR = _config['AR']
    L = _config['L']
    bar_width = _config['bar_width']

    mesh = RectangleMesh(Point(-L,-L),Point(L,L), _config['res'], _config['res'], 'left/right')
    for i in range(_config['res_iterations']):
        cell_markers = MeshFunction("bool", mesh, 2)
        cell_markers.set_all(False)
        for cell in cells(mesh):
            mp = cell.midpoint().array()
            if mp[0]>-1.2 and mp[0]<1.2 and  mp[1]>-1.2 and mp[1]<1.2:
                cell_markers[cell]=True
        mesh = refine(mesh, cell_markers)

    # Build function space
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), _config['degree_fem_velocity']) ## For the velocity
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), _config['degree_fem_pressure']) ## For the pressure
    TH = P2 * P1 ## Taylor-Hood elements
    W = FunctionSpace(mesh, TH) ## on the mesh

    # Boundaries
    velocity_to_center = Expression(("-x[0]*v", "-x[1]*v"), v = Constant(_config['v_scale']), degree=2)
    bcs = []
    bcs.append(DirichletBC(W.sub(0), velocity_to_center, lambda x, on_boundary: cross(x, on_boundary, AR, bar_width)))

    # No-slip boundary conditions
    if _config['no_slip_top_bottom']:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, lambda x, on_boundary: top_bottom(x, on_boundary, L)))
    if _config['no_slip_left_right']:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, lambda x, on_boundary: left_right(x, on_boundary, L)))
    if _config['no_slip_center_size']>0:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, lambda x, on_boundary: inner_noslip(x, on_boundary, AR, _config['no_slip_center_size'])))

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
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
            _config['res_iterations'] += 1
            print(f"Boundary conditions cannot be implemented. Retrying with a higher adaptivity")
            return solve_rectangle(_config)
        else:
            print('There was an error in assemble_system. The output of FeniCS:')
            print(outstring)

    # Create Krylov solver and AMG preconditioner
    solver = KrylovSolver(_config['krylov_method'], "amg")

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    # Solve
    U = Function(W)
    solver.solve(U.vector(), bb)

    # Get sub-functions
    u, p = U.split()
    return u,p
