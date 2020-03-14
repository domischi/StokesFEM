from dolfin import *
import mshr
import sys
from capture_cpp_cout import capture_cpp_cout

def solve_rectangle(_config):
    AR = _config['AR']
    L = _config['L']
    bar_width = _config['bar_width']

    mesh = RectangleMesh(Point(-L,-L),Point(L,L), _config['res'], _config['res'], 'left/right')
#mesh = mshr.generate_mesh(mshr.Rectangle(Point(-L,-L),Point(L,L)), res)

# Build function space
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), _config['degree_fem_velocity']) ## For the velocity
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), _config['degree_fem_pressure']) ## For the pressure
    TH = P2 * P1 ## Taylor-Hood elements
    W = FunctionSpace(mesh, TH) ## on the mesh

# Boundaries
    def diagonal_ur(x, on_boundary): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]-AR) < bar_width and x[0]>0 and x[1]>0 and x[0]< 1 and x[1]< AR ## upper right quadrant
    def diagonal_ul(x, on_boundary): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]+AR) < bar_width and x[0]<0 and x[1]>0 and x[0]>-1 and x[1]< AR ## upper left  quadrant
    def diagonal_dr(x, on_boundary): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]+AR) < bar_width and x[0]>0 and x[1]<0 and x[0]< 1 and x[1]>-AR ## down  right quadrant
    def diagonal_dl(x, on_boundary): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]-AR) < bar_width and x[0]<0 and x[1]<0 and x[0]>-1 and x[1]>-AR ## down  left  quadrant
    def left_right(x, on_boundary): return x[0] > L - DOLFIN_EPS or x[0] < -L+DOLFIN_EPS
    def top_bottom(x, on_boundary): return x[1] > L - DOLFIN_EPS or x[1] < -L+DOLFIN_EPS

    velocity_to_center = Expression(("-x[0]", "-x[1]"), degree=2)
    bcs = []
    bcs.append(DirichletBC(W.sub(0), velocity_to_center, diagonal_ur))
    bcs.append(DirichletBC(W.sub(0), velocity_to_center, diagonal_ul))
    bcs.append(DirichletBC(W.sub(0), velocity_to_center, diagonal_dr))
    bcs.append(DirichletBC(W.sub(0), velocity_to_center, diagonal_dl))

    # No-slip boundary conditions
    if _config['no_slip_top_bottom']:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, top_bottom))
    if _config['no_slip_left_right']:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, left_right))

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0))
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
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
            print("Boundary conditions broke. There are no nodes inside the Dirichlet boundary region.")
            sys.exit()
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
