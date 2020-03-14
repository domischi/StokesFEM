# Tutorial from: https://fenicsproject.org/docs/dolfin/2019.1.0/python/demos/stokes-iterative/demo_stokes-iterative.py.html
from dolfin import *
import mshr
import sys
from illustration import plot_fluid
from capture_cpp_cout import capture_cpp_cout
import numpy as np
import sys

AR = 1.
L = 3
res = 50
bar_width=.1
krylov_method = "minres" ## alternatively use tfqrm

mesh = RectangleMesh(Point(-3,-3),Point(3,3), res,res, 'left/right')
#mesh = mshr.generate_mesh(mshr.Rectangle(Point(-L,-L),Point(L,L)), res)

# Build function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) ## For the velocity
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) ## For the pressure
TH = P2 * P1 ## Taylor-Hood elements
W = FunctionSpace(mesh, TH) ## on the mesh

# Boundaries
def diagonal_to_center_ur(x, on_boundary): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]-AR) < bar_width and x[0]>0 and x[1]>0 and x[0]< 1 and x[1]< AR ## upper right quadrant
def diagonal_to_center_ul(x, on_boundary): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]+AR) < bar_width and x[0]<0 and x[1]>0 and x[0]>-1 and x[1]< AR ## upper left  quadrant
def diagonal_to_center_dr(x, on_boundary): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]+AR) < bar_width and x[0]>0 and x[1]<0 and x[0]< 1 and x[1]>-AR ## down  right quadrant
def diagonal_to_center_dl(x, on_boundary): return abs(x[0])>DOLFIN_EPS and abs(x[1]/x[0]-AR) < bar_width and x[0]<0 and x[1]<0 and x[0]>-1 and x[1]>-AR ## down  left  quadrant

velocity_to_center = Expression(("-x[0]", "-x[1]"), degree=2)
bc0 = DirichletBC(W.sub(0), velocity_to_center, diagonal_to_center_ur)
bc1 = DirichletBC(W.sub(0), velocity_to_center, diagonal_to_center_ul)
bc2 = DirichletBC(W.sub(0), velocity_to_center, diagonal_to_center_dr)
bc3 = DirichletBC(W.sub(0), velocity_to_center, diagonal_to_center_dl)

# No-slip boundary condition for velocity
#noslip = Constant((0.0, 0.0, 0.0))
#bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

# Collect boundary conditions
bcs = [bc0, bc1, bc2, bc3]

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
solver = KrylovSolver(krylov_method, "amg")

# Associate operator (A) and preconditioner matrix (P)
solver.set_operators(A, P)

# Solve
U = Function(W)
solver.solve(U.vector(), bb)

# Get sub-functions
u, p = U.split()

plot_fluid(u, L=L, plot_type='quiver')
