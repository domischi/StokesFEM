# Tutorial from: https://fenicsproject.org/docs/dolfin/2019.1.0/python/demos/stokes-iterative/demo_stokes-iterative.py.html
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from dolfin import *
import mshr
import sys
from illustration import plot_fluid
from capture_cpp_cout import capture_cpp_cout
from rectangle import solve_rectangle
import numpy as np
import sys

ex = Experiment('Diagonal FEM')
ex.observers.append(MongoObserver.create())
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    ## Geometry parameters
    AR = 1.
    L = 3
    res = 50
    bar_width=.1
    krylov_method = "minres" ## alternatively use tfqrm ## TODO use it in the integrator

    ## Fluid parameters
    mu=1. ## TODO use it in the integrator


@ex.automain
def main(_config):
    AR = _config['AR']
    L = _config['L']
    res = _config['res']
    bar_width = _config['bar_width']
    u, p = solve_rectangle(AR,L,res,bar_width)
    plot_fluid(u, L=L, plot_type='quiver')
