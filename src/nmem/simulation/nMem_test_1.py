# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:38:40 2023

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
from phidl.device_layout import Polygon
from tdgl.geometry import box
from tdgl.visualization.animate import create_animation

os.environ["OPENBLAS_NUM_THREADS"] = "0"


plt.rcParams["figure.figsize"] = (5, 4)

# sys.path.append(r'C:\Users\omedeiro\Documents\GitHub\py-tdgl\tdgl')


def make_video_from_solution(
    solution,
    quantities=("order_parameter", "phase"),
    fps=20,
    figsize=(5, 4),
):
    """Generates an HTML5 video from a tdgl.Solution."""
    with tdgl.non_gui_backend():
        with h5py.File(solution.path, "r") as h5file:
            anim = create_animation(
                h5file,
                quantities=quantities,
                fps=fps,
                figure_kwargs=dict(figsize=figsize),
            )
            video = anim.to_html5_video()
        return HTML(video)


tempdir = tempfile.TemporaryDirectory()

plt.close()

length_units = "um"
# Material parameters
xi = 0.0062
# xi = 0.0062

london_lambda = 0.2
d = 0.01

mu0 = 1.256637062120000e-06
sigma = 1 / (2.5)  # (uΩ-cm)^-1
h = 6.626070150000000e-34
e = 1.602176634000000e-19
phi0 = h / (2 * e)
# UNITS
tau0 = mu0 * sigma * london_lambda**2
B0 = phi0 / (2 * np.pi * xi**2)
A0 = xi * B0
J0 = (4 * xi * B0) / (mu0 * london_lambda**2)
K0 = J0 * d  # uA/um
V0 = xi * J0 / sigma

layer = tdgl.Layer(
    coherence_length=xi,
    london_lambda=london_lambda,
    thickness=d,
    conductivity=0.4,
    gamma=23.8,
)


mem = qg.memory_v4()

pts = mem.polygons[0]
p = pts.polygons[0]
pout1 = p[1:305]
pout = np.append(pout1, p[-2:], axis=0)
pin = p[305:-2]

pp = Polygon(pout, 0, 0, None)


left_path = np.zeros((10, 2))
left_path[:, 0] = np.linspace(-0.1, 0.1, 10)
left_path[:, 1] = np.ones((10,)) * 0.3

right_path = np.zeros((10, 2))
right_path[:, 0] = np.linspace(2.7, 3.1, 10)
right_path[:, 1] = np.ones((10,)) * 0.3

x_positions = np.linspace(-0.7, 3.3, 50)
y_positions = np.linspace(-3.75, 3.75, 50)
field_positions = np.zeros((50 * 50, 2))
field_positions[:, 0] = np.tile(x_positions, 50)
field_positions[:, 1] = np.repeat(y_positions, 50)

# %%
plt.close()
film = tdgl.Polygon("film", points=pout).buffer(0).resample(601)

round_hole = tdgl.Polygon("center", points=pin).buffer(0).resample(601)

source = tdgl.Polygon("source", points=box(2.1, 1 / 10)).translate(dx=0.5, dy=3.4)

drain = tdgl.Polygon("drain", points=box(2.1, 1 / 10)).translate(dx=0.5, dy=-3.4)

probe_points = [(0.5, 3), (0.5, -3)]


device = tdgl.Device(
    "weak_link",
    layer=layer,
    film=film,
    holes=[round_hole],
    terminals=[source, drain],
    probe_points=probe_points,
    length_units=length_units,
)


fig, ax = device.draw()

device.make_mesh(max_edge_length=xi * 10)
fig, ax = device.plot(mesh=True, legend=False)


# %%


tables.file._open_files.close_all()
strnow = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

dirname = r"G:\My Drive\___Projects\_physics\py-tdgl\nMem\test"
simName = "IV_seed"
path = os.path.join(dirname, simName, strnow)

filename = f"nMem_{strnow}"

applied_current = 2000
stime = 100

prev_sol = None

N = 11
cr = np.linspace(0, applied_current, N)
cr = np.append(cr, np.flipud(cr[0:-1]))
n = len(cr)
left_current = np.zeros((n, 1))
right_current = np.zeros((n, 1))
left_ncurrent = np.zeros((n, 1))
right_ncurrent = np.zeros((n, 1))
voltage = np.zeros((n, 1))
total_current = np.ones((n, 1))


for i, current in enumerate(cr):
    full_name = os.path.join(path, f"{i}current_{int(current)}uA")
    options = tdgl.SolverOptions(
        # Allow some time to equilibrate before saving data.
        skip_time=500,
        solve_time=stime,
        output_file=full_name + ".h5",
        field_units="mT",
        current_units="uA",
        save_every=50,
    )

    nmem_solution = tdgl.solve(
        device,
        options,
        # terminal_currents must satisfy current conservation, i.e.,
        # sum(terminal_currents.values()) == 0.
        applied_vector_potential=0,
        terminal_currents=dict(source=current, drain=-current),
        seed_solution=prev_sol,
    )

    prev_sol = nmem_solution

    fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(16, 9), layout="constrained")

    fig, ax = nmem_solution.plot_currents(
        dataset="supercurrent",
        # figsize=(6, 3),
        streamplot=False,
        ax=axs[0, 0],
    )
    ax.plot(left_path[:, 0], left_path[:, 1], color="C0")
    ax.plot(right_path[:, 0], right_path[:, 1], color="C1")

    fig, ax = nmem_solution.plot_currents(
        dataset="normal_current",
        # figsize=(6, 3),
        streamplot=False,
        ax=axs[0, 1],
    )
    ax.plot(left_path[:, 0], left_path[:, 1], color="C0")
    ax.plot(right_path[:, 0], right_path[:, 1], color="C1")

    fig, ax = nmem_solution.plot_order_parameter(ax=axs[0, 2:4])

    fig, ax = nmem_solution.plot_field_at_positions(
        field_positions, zs=0.1, ax=axs[1, 3], vmin=-3, vmax=3
    )

    left_current[i] = nmem_solution.current_through_path(
        left_path, with_units=False, dataset="supercurrent"
    )
    left_ncurrent[i] = nmem_solution.current_through_path(
        left_path, with_units=False, dataset="normal_current"
    )

    right_current[i] = nmem_solution.current_through_path(
        right_path, with_units=False, dataset="supercurrent"
    )
    right_ncurrent[i] = nmem_solution.current_through_path(
        right_path, with_units=False, dataset="normal_current"
    )

    total_current[i] = (
        left_current[i] + left_ncurrent[i] + right_current[i] + right_ncurrent[i]
    )
    if total_current[i] == 0:
        total_current[i] = 1

    ax = axs[1, 0]
    ax.plot(
        cr[0 : i + 1],
        left_current[0 : i + 1] / total_current[0 : i + 1],
        color="C0",
        marker="o",
        label="left",
    )
    ax.plot(
        cr[0 : i + 1],
        right_current[0 : i + 1] / total_current[0 : i + 1],
        color="C1",
        marker="o",
        label="right",
    )
    ax.legend()
    ax.set_title("Supercurrent")
    ax.set_ylabel("$I_{super}/I_{total}$")
    ax.set_xlabel("$I_{applied}$ [$\\mu$ A]")
    ax.set_ylim([0, 1])

    ax = axs[1, 1]
    ax.plot(
        cr[0 : i + 1],
        left_ncurrent[0 : i + 1] / total_current[0 : i + 1],
        color="C0",
        marker="o",
        label="left",
    )
    ax.plot(
        cr[0 : i + 1],
        right_ncurrent[0 : i + 1] / total_current[0 : i + 1],
        color="C1",
        marker="o",
        label="right",
    )
    ax.legend()
    ax.set_title("Normal current")
    ax.set_ylabel("$I_{norm}$/$I_{total}$")
    ax.set_xlabel("$I_{applied}$ [$\\mu$ A]")
    ax.set_ylim([0, 1])

    ax = axs[1, 2]
    voltage[i] = nmem_solution.dynamics.voltage()[-1]
    ax.plot(total_current[0 : i + 1], voltage[0 : i + 1], marker="o")
    ax.set_xlabel("$I_{total}$ [$\\mu$ A]")
    ax.set_ylabel("$V$ [$V_0$]")

    fig.suptitle(full_name)
    fig.savefig(full_name + "_fig")
    plt.close(fig)
# plt.savefig(args, kwargs)

MAKE_ANIMATIONS = False

tempdir.cleanup()


if MAKE_ANIMATIONS:
    nMem_video = make_video_from_solution(
        nmem_solution,
        quantities=["order_parameter", "phase", "scalar_potential"],
        figsize=(6.5, 4),
    )
    display(nMem_video)

    with open(path + "\data_" + strnow + ".html", "w") as file:
        file.write(nMem_video.data)
