# Tutorial from: https://fenicsproject.org/docs/dolfin/2019.1.0/python/demos/stokes-iterative/demo_stokes-iterative.py.html
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from dolfin import *
import pickle as pkl
import mshr
import sys
import time
from illustration import sample_velocity, plot_fluid, plot_pressure
from capture_cpp_cout import capture_cpp_cout
from rectangle import solve_rectangle
import numpy as np
import sys

ex = Experiment('Diagonal FEM')
ex.observers.append(MongoObserver.create())
SETTINGS.CONFIG.READ_ONLY_CONFIG = False ## Needed if the mesh needs to be refined to define BC
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    ## Numerical parameters:
    degree_fem_velocity = 2 # That's Taylor-Hood
    degree_fem_pressure = degree_fem_velocity-1

    ## Geometry parameters
    AR = .8
    L = 8
    res = 50
    res_iterations = 2
    bar_width=.1
    krylov_method = "minres" ## alternatively use tfqrm
    v_scale = 1.

    ## Boundary
    diagonal_bc = False
    no_slip_top_bottom = True
    no_slip_left_right = True
    no_penetration_top_bottom = False
    no_penetration_left_right = False
    no_slip_center_size = .5

    ## Force Fields
    rectangular_ff = False
    diagonal_ff = True
    Fscale = 1

    ## Fluid parameters
    mu=1.

    ## Plotting parameters
    Lplot = 2
    plot_res = 32
    plot_type = 'quiver'
    color_scheme = 'vabs'
    plot_rectangle = True
    plot_cross = True

    ## Saving behavior
    save_sampled_fluid_field = True


@ex.automain
def main(_config):
    u, p = solve_rectangle(_config)
    vals = sample_velocity(u, _config)
    if _config['save_sampled_fluid_field']:
        pickle_filename = f'/tmp/fem-vals-{int(time.time())}.pkl'
        with open(pickle_filename, 'wb') as f:
            pkl.dump(vals,f)
        ex.add_artifact(pickle_filename)
    figure_filename = plot_fluid(u, _config, already_sampled_values=vals)
    ex.add_artifact(figure_filename)
    figure_filename = plot_pressure(p, _config)
    ex.add_artifact(figure_filename)
