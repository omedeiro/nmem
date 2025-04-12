import argparse
import glob
import os
from pathlib import Path
import numpy as np
from tdgl.solution.solution import Solution
import h5py
from nmem.simulation.pytdgl.sim.util import (
    get_current_through_path,
    make_animation_from_solution,
    find_latest_result_file,
    make_field_animation,
)
from matplotlib import pyplot as plt
from IPython.display import HTML, display
import tdgl
from tdgl.visualization.animate import create_animation


def make_video_from_solution(
    solution, quantities=("order_parameter", "phase"), fps=20, figsize=(5, 4)
):
    with tdgl.non_gui_backend():
        with h5py.File(solution.path, "r") as h5file:
            anim = create_animation(
                h5file,
                quantities=quantities,
                fps=fps,
                figure_kwargs=dict(figsize=figsize),
            )
            return HTML(anim.to_html5_video())


def find_latest_result_file(output_root: str = "output") -> str:
    """Finds the most recently modified .h5 file in the output directory tree."""
    files = glob.glob(f"{output_root}/**/*.h5", recursive=True)
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {output_root}")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


# --- Determine file to load ---
file_to_load = find_latest_result_file()
print(f"Loading: {file_to_load}")

sol = Solution.from_hdf5(file_to_load)

# --- Paths for current extraction ---
main_path = np.column_stack((np.linspace(-0.5, 1.5, 100), np.full(100, 2.5)))
left_path = np.column_stack((np.linspace(-0.1, 0.1, 100), np.full(100, 0.3)))
right_path = np.column_stack((np.linspace(2.7, 3.1, 100), np.full(100, 0.3)))

# --- Supercurrent plot ---
fig, ax = sol.plot_currents(dataset="supercurrent", streamplot=True)

ax.plot(left_path[:, 0], left_path[:, 1], color="C0", lw=1)
ax.plot(right_path[:, 0], right_path[:, 1], color="C1", lw=1)
ax.plot(main_path[:, 0], main_path[:, 1], color="C2", lw=1)
fig.suptitle("Supercurrent")
plt.show()
# --- Normal current plot ---
fig, ax = sol.plot_currents(dataset="normal_current", streamplot=True)
ax.plot(left_path[:, 0], left_path[:, 1], color="C0", lw=1)
ax.plot(right_path[:, 0], right_path[:, 1], color="C1", lw=1)
fig.suptitle("Normal Current")
plt.show()

# --- Order parameter ---
sol.plot_order_parameter()


# --- Magnetic field ---
x_pos = np.linspace(-0.7, 3.3, 50)
y_pos = np.linspace(-3.75, 3.75, 50)
field_pos = np.column_stack((np.tile(x_pos, 50), np.repeat(y_pos, 50)))
sol.plot_field_at_positions(field_pos, zs=0.01, vmin=-0.3, vmax=0.3)

# --- Extract currents and print ---
applied_supercurrent = get_current_through_path(sol, main_path, dataset="supercurrent")
applied_normal = get_current_through_path(sol, main_path, dataset="normal_current")
print(f"Applied supercurrent: {applied_supercurrent}")
print(f"Applied normal current: {applied_normal}")


left_supercurrent = get_current_through_path(sol, left_path, dataset="supercurrent")
right_supercurrent = get_current_through_path(sol, right_path, dataset="supercurrent")
left_normal = get_current_through_path(sol, left_path, dataset="normal_current")
right_normal = get_current_through_path(sol, right_path, dataset="normal_current")

print(f"Left supercurrent: {left_supercurrent}")
print(f"Right supercurrent: {right_supercurrent}")
print(f"Left normal current: {left_normal}")
print(f"Right normal current: {right_normal}")

# --- Make animation ---
output_path = os.path.dirname(file_to_load)
tag = os.path.basename(file_to_load).split(".")[0]
video_html= make_animation_from_solution(
    sol,
    output_path,
    tag,
    quantities=("order_parameter", "phase", "supercurrent", "normal_current"),
)
# --- Save animation as HTML ---
html_file = os.path.join(output_path, f"data_{tag}.html")
with open(html_file, "w") as file:
    file.write(video_html.data)


# output_path = os.path.dirname(file_to_load)
# tag = os.path.basename(file_to_load).split(".")[0]

make_field_animation(sol, output_path=output_path, tag=tag)
