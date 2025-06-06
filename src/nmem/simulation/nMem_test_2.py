# -*- coding: utf-8 -*-
"""
Refactored TDGL Simulation for Superconducting Nanowire Memory
Created on Tue Nov 28 19:21:58 2023
@author: omedeiro
"""

import os
import tempfile
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import qnngds.geometry as qg
import tables
import tdgl
from IPython.display import HTML, display
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import explain_validity
from tdgl.geometry import box
from tdgl.visualization.animate import create_animation

# Plot settings and environment variables
os.environ["OPENBLAS_NUM_THREADS"] = "0"
plt.rcParams["figure.figsize"] = (5, 4)

# Physical constants
length_units = "um"
XI = 0.0062
LONDONL = 0.2
D = 0.01
MU0 = 4 * np.pi * 1e-7
RESISTIVITY = 2.5
SIGMA = 1 / RESISTIVITY
H = 6.62607015e-34
E = 1.602176634e-19
PHI0 = H / (2 * E)
TAU0 = MU0 * SIGMA * LONDONL**2
B0 = PHI0 / (2 * np.pi * XI**2)
A0 = XI * B0
J0 = (4 * XI * B0) / (MU0 * LONDONL**2)
K0 = J0 * D
V0 = XI * J0 / SIGMA


def fix_polygon(coords: np.ndarray) -> ShapelyPolygon:
    poly = ShapelyPolygon(coords)
    print("Original validity:", explain_validity(poly))
    if poly.is_valid:
        return poly
    fixed = poly.buffer(0)
    if not fixed.is_valid:
        print("Still invalid after fix:", explain_validity(fixed))
    return fixed


def make_video_from_solution(solution, quantities=("order_parameter", "phase"), fps=20, figsize=(5, 4)):
    with tdgl.non_gui_backend():
        with h5py.File(solution.path, "r") as h5file:
            anim = create_animation(h5file, quantities=quantities, fps=fps, figure_kwargs=dict(figsize=figsize))
            return HTML(anim.to_html5_video())


def make_animation_from_solution(solution, output_path, tag, quantities=("order_parameter", "phase"), fps=20):
    video_html = make_video_from_solution(solution, quantities=quantities, fps=fps)
    display(video_html)
    html_file = os.path.join(output_path, f"data_{tag}.html")
    with open(html_file, "w") as file:
        file.write(video_html.data)


def make_device(xi=XI, d=D, london_lambda=LONDONL, conductivity=SIGMA, gamma=23.8):
    layer = tdgl.Layer(
        coherence_length=xi,
        london_lambda=london_lambda,
        thickness=d,
        conductivity=conductivity,
        gamma=gamma,
    )
    mem = qg.memory_v4()
    p = mem.polygons[0].polygons[0]
    pout = np.vstack((p[:149], p[-2:], p[:1]))
    pin = np.vstack((p[150:-2], p[150:151]))

    film = tdgl.Polygon("film", points=pout)
    hole = tdgl.Polygon("center", points=pin).buffer(0)
    source = tdgl.Polygon("source", points=box(2.1, 0.1)).translate(dx=0.5, dy=3.4)
    drain = tdgl.Polygon("drain", points=box(2.1, 0.1)).translate(dx=0.5, dy=-3.4)
    probes = [(0.5, 3), (0.5, -3)]

    return tdgl.Device(
        "weak_link",
        layer=layer,
        film=film,
        holes=[hole],
        terminals=[source, drain],
        probe_points=probes,
        length_units=length_units,
    )


def get_current_through_path(solution, path, dataset="supercurrent", with_units=True):
    return solution.current_through_path([tuple(coord) for coord in path], with_units=with_units, dataset=dataset)


def run_simulation(device, current, stime, path, prev_sol=None):
    options = tdgl.SolverOptions(
        skip_time=500,
        solve_time=stime,
        output_file=os.path.join(path, f"current_{int(current)}uA.h5"),
        field_units="mT",
        current_units="uA",
        save_every=50,
    )
    return tdgl.solve(
        device,
        options,
        applied_vector_potential=0,
        terminal_currents={"source": current, "drain": -current},
        seed_solution=prev_sol,
    )


if __name__ == "__main__":
    tempdir = tempfile.TemporaryDirectory()
    tables.file._open_files.close_all()
    plt.close()

    # Set up output directory
    timestamp = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    output_root = r"G:\My Drive\___Projects\_physics\py-tdgl\nMem\test"
    sim_name = "IV_seed"
    output_path = os.path.join(output_root, sim_name, timestamp)
    os.makedirs(output_path, exist_ok=True)

    # Make device
    device = make_device()
    device.draw()
    device.make_mesh(max_edge_length=XI * 10)
    device.plot(mesh=True, legend=False)

    # Simulation parameters
    current = 2000  # uA
    stime = 100  # ps
    prev_sol = None

    # Measurement paths
    left_path = np.column_stack((np.linspace(-0.1, 0.1, 10), np.full(10, 0.3)))
    right_path = np.column_stack((np.linspace(2.7, 3.1, 10), np.full(10, 0.3)))

    # Run simulation
    sol = run_simulation(device, current=current, stime=stime, path=output_path, prev_sol=prev_sol)

    # Plot results
    sol.plot_currents(dataset="supercurrent", streamplot=False)[1].plot(left_path[:, 0], left_path[:, 1], color="C0")
    sol.plot_currents(dataset="supercurrent", streamplot=False)[1].plot(right_path[:, 0], right_path[:, 1], color="C1")
    sol.plot_currents(dataset="normal_current", streamplot=False)[1].plot(left_path[:, 0], left_path[:, 1], color="C0")
    sol.plot_currents(dataset="normal_current", streamplot=False)[1].plot(right_path[:, 0], right_path[:, 1], color="C1")
    sol.plot_order_parameter()

    field_positions = np.column_stack((
        np.tile(np.linspace(-0.7, 3.3, 50), 50),
        np.repeat(np.linspace(-3.75, 3.75, 50), 50),
    ))
    sol.plot_field_at_positions(field_positions, zs=0.1, vmin=-3, vmax=3)

    # Extract currents
    left_current = get_current_through_path(sol, left_path, "supercurrent")
    right_current = get_current_through_path(sol, right_path, "supercurrent")
    left_ncurrent = get_current_through_path(sol, left_path, "normal_current")
    right_ncurrent = get_current_through_path(sol, right_path, "normal_current")

    # Optional: animation
    # make_animation_from_solution(sol, output_path, timestamp)

    tempdir.cleanup()
