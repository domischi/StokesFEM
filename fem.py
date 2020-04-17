from dolfin import *

def left_right  (x, on_boundary, L): return x[0] > L * (1-DOLFIN_EPS) or x[0] < L * (-1+DOLFIN_EPS)
def top_bottom  (x, on_boundary, L): return x[1] > L * (1-DOLFIN_EPS) or x[1] < L * (-1+DOLFIN_EPS)

def get_function_space(_config, mesh):
    # Build function space
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), _config['degree_fem_velocity']) ## For the velocity
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), _config['degree_fem_pressure']) ## For the pressure
    TH = P2 * P1 ## Taylor-Hood elements
    return FunctionSpace(mesh, TH) ## on the mesh

def get_general_bcs(_config, W):
    bcs = []
    L = _config['L']
    # No-slip boundary conditions
    if _config['no_slip_top_bottom']:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, lambda x, on_boundary: top_bottom(x, on_boundary, L)))
    if _config['no_slip_left_right']:
        noslip = Constant((0.0, 0.0))
        bcs.append(DirichletBC(W.sub(0), noslip, lambda x, on_boundary: left_right(x, on_boundary, L)))
    # No penetration into wall
    if (not _config['no_slip_top_bottom']) and _config['no_penetration_top_bottom']:
        bcs.append(DirichletBC(W.sub(0).sub(1), Constant(0.), lambda x, on_boundary: top_bottom(x, on_boundary, L)))
    if (not _config['no_slip_left_right']) and _config['no_penetration_left_right']:
        bcs.append(DirichletBC(W.sub(0).sub(0), Constant(0.), lambda x, on_boundary: left_right(x, on_boundary, L)))
    return bcs

def solve_stokes(_config, assemble_system):
    mesh, W, A, P, bb = assemble_system(_config)
    # Create Krylov solver and AMG preconditioner
    solver = KrylovSolver(_config['krylov_method'], "amg")
    solver.parameters.update( { 'absolute_tolerance' : 1e-9, 'relative_tolerance' : 1e-9, 'maximum_iterations' : 1000, })

    # Associate operator (A) and preconditioner matrix (P)
    solver.set_operators(A, P)

    # Solve
    U = Function(W)
    solver.solve(U.vector(), bb)

    # Get sub-functions
    u, p = U.split()
    return u, p, mesh
