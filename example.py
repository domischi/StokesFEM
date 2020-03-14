# Tutorial from: https://fenicsproject.org/docs/dolfin/2019.1.0/python/demos/stokes-iterative/demo_stokes-iterative.py.html
from dolfin import *
import mshr
import sys
from illustration import plot_fluid
from capture_cpp_cout import capture_cpp_cout
from rectangle import solve_rectangle
import numpy as np
import sys

AR = 1.
L = 3
res = 50
bar_width=.1
krylov_method = "minres" ## alternatively use tfqrm


u, p = solve_rectangle(AR,L,res,bar_width)

plot_fluid(u, L=L, plot_type='quiver')
