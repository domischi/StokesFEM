# Tutorial from: https://fenicsproject.org/docs/dolfin/2019.1.0/python/demos/stokes-iterative/demo_stokes-iterative.py.html
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from dolfin import *
import json
import mshr
import sys
import time
from illustration import sample_velocity, plot_fluid, plot_pressure
from capture_cpp_cout import capture_cpp_cout
from geometries.rectangle import assemble_rectangular_system
from fem import solve_stokes
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
    Geometry = 'rectangle'
    AR = .4
    L = 5
    res = 50
    initial_res_iterations = 3
    initial_res_iterations = 5
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
    diagonal_ff = False
    corner_ff = True
    Fscale = 1

    ## Fluid parameters
    mu=1.

    ## Plotting parameters
    Lplot = 3
    plot_res = 32
    plot_type = 'quiver'
    color_scheme = 'vabs'
    plot_active = False
    plot_cross = False
    plot_corner = False

    ## Saving behavior
    save_sampled_fluid_field = True
    save_mesh = True


@ex.automain
def main(_config):
    if _config['Geometry'] == 'rectangle':
        assemble_system = assemble_rectangular_system
        title = f"AR={_config['AR']:.3f}"
    elif _config['Geometry'] == 'hexagon':
        assemble_system = assemble_hexagon_system
        title = f"Hexagon"
    else:
        raise RuntimeError(f"Did not recognize geometry {_config['Geometry']}. Aborting.")

    u, p, mesh = solve_stokes(_config, assemble_system)
    if _config['save_mesh']:
        filename = f'/tmp/fem-mesh-{int(time.time())}.json'
        with open(filename, 'w') as f:
            json.dump(mesh.coordinates().tolist(),f)
        ex.add_artifact(filename)

    X,Y,U,V = sample_velocity(u, _config)
    if _config['save_sampled_fluid_field']:
        filename = f'/tmp/fem-vals-{int(time.time())}.json'
        Xp,Yp,Up,Vp = sample_velocity(u, _config, L=_config['Lplot'])
        with open(filename, 'w') as f:
            d = {'X' : X .tolist(), 'Y' : Y .tolist(), 'U' : U .tolist(), 'V' : V .tolist(),
                 'Xp': Xp.tolist(), 'Yp': Yp.tolist(), 'Up': Up.tolist(), 'Vp': Vp.tolist()}
            json.dump(d,f)
        ex.add_artifact(filename)
    figure_filename = plot_fluid(u, _config, already_sampled_values=(X,Y,U,V), title = title)
    ex.add_artifact(figure_filename)
    figure_filename = plot_pressure(p, _config)
    ex.add_artifact(figure_filename)
