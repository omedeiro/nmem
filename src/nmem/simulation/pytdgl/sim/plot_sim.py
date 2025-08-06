"""
This script plots the results of a transient trangular waveform with both positive and negative amplitudes.
Both amplitudes are plotted to visualize the opposite circulating current.
"""

import os

import h5py
import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
import tdgl
from IPython.display import HTML
from matplotlib import pyplot as plt
from tdgl.solution.solution import Solution
from tdgl.visualization.animate import create_animation

from nmem.simulation.pytdgl.sim.util import (
    get_current_through_path,
)


def set_inter_font():
    if os.name == "nt":  # Windows
        font_path = r"C:\Users\ICE\AppData\Local\Microsoft\Windows\Fonts\Inter-VariableFont_opsz,wght.ttf"
    elif os.name == "posix":
        font_path = "/home/omedeiro/Inter-VariableFont_opsz,wght.ttf"
    else:
        font_path = None

    if font_path and os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        mpl.rcParams["font.family"] = "Inter"


set_inter_font()
plt.rcParams.update(
    {
        # "figure.figsize": [width, height],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "Inter",
        "lines.markersize": 3,
        "lines.linewidth": 1.2,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)


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


def plot_supercurrent(
    ax: plt.Axes, sol: Solution, solve_step: int = 200, streamplot=True, **kwargs
):
    sol.solve_step = solve_step
    sol.plot_currents(dataset="supercurrent", streamplot=streamplot, ax=ax, **kwargs)

    return ax


def plot_normalcurrent(ax: plt.Axes, sol: Solution, solve_step: int = 200):
    sol.solve_step = solve_step
    sol.plot_currents(dataset="normal_current", streamplot=True, ax=ax)
    return ax


def plot_order_parameter(ax: plt.Axes, sol: Solution):
    sol.plot
    return ax


def plot_curlines(ax: plt.Axes, path: np.ndarray, color: str):
    ax.plot(path[:, 0], path[:, 1], color=color, lw=1)
    return ax


def plot_magnetic_field(
    ax: plt.Axes,
    sol: Solution,
    field_pos: np.ndarray,
    zs: float = 0.01,
    solve_step: int = 10,
    x_pos: np.ndarray = None,
    y_pos: np.ndarray = None,
):
    sol.solve_step = solve_step

    # If x_pos and y_pos are not provided, infer them from field_pos
    if x_pos is None or y_pos is None:
        # Assume field_pos is a meshgrid flattened
        x_pos = np.unique(field_pos[:, 0])
        y_pos = np.unique(field_pos[:, 1])

    Bz0 = sol.field_at_position(field_pos, zs=zs).reshape(len(y_pos), len(x_pos))
    vmin = float(np.nanmin(Bz0).magnitude)
    vmax = float(np.nanmax(Bz0).magnitude)

    max_val = np.max(np.abs([vmin, vmax]))
    im = ax.imshow(
        Bz0,
        extent=[x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()],
        origin="lower",
        cmap="RdBu_r",
        vmin=-max_val,
        vmax=max_val,
    )
    return im


def get_currents():
    # --- Extract currents and print ---
    applied_supercurrent = get_current_through_path(
        sol, main_path, dataset="supercurrent"
    )
    applied_normal = get_current_through_path(sol, main_path, dataset="normal_current")
    left_supercurrent = get_current_through_path(sol, left_path, dataset="supercurrent")
    right_supercurrent = get_current_through_path(
        sol, right_path, dataset="supercurrent"
    )
    left_normal = get_current_through_path(sol, left_path, dataset="normal_current")
    right_normal = get_current_through_path(sol, right_path, dataset="normal_current")
    return (
        applied_supercurrent,
        applied_normal,
        left_supercurrent,
        right_supercurrent,
        left_normal,
        right_normal,
    )


def get_currents_vs_time(
    sol: Solution,
    left_path: np.ndarray,
    right_path: np.ndarray,
    dataset: str = "supercurrent",
):
    """Extract current through left and right paths for all time steps."""
    times = []
    left_currents = []
    right_currents = []

    # Get the range of available time steps
    start_step, end_step = sol.data_range

    for step in range(start_step, end_step + 1):
        sol.solve_step = step

        # Get the time for this step
        time = (
            sol.times[step] if hasattr(sol, "times") else step * sol.options.save_every
        )
        times.append(time)

        # Get currents through each path
        left_current = get_current_through_path(
            sol, left_path, dataset=dataset, with_units=True
        )
        right_current = get_current_through_path(
            sol, right_path, dataset=dataset, with_units=True
        )

        left_currents.append(left_current)
        right_currents.append(right_current)

    return times, left_currents, right_currents


def plot_current_time_series(
    ax: plt.Axes, times: list, left_currents: list, right_currents: list
):
    """Plot current vs time for left and right branches."""
    # Convert times to numpy array if it's not already
    times = np.array(times)

    # Convert to scalar values if they have units
    if len(left_currents) > 0 and hasattr(left_currents[0], "magnitude"):
        left_vals = [curr.magnitude for curr in left_currents]
        right_vals = [curr.magnitude for curr in right_currents]
        ylabel = f"Current [{left_currents[0].units}]"
    else:
        left_vals = left_currents
        right_vals = right_currents
        ylabel = "Current [μA]"

    left_vals = np.array(left_vals)
    right_vals = np.array(right_vals)

    ax.plot(times, left_vals, label="Left branch", color="blue", linewidth=1.5)
    ax.plot(times, right_vals, label="Right branch", color="red", linewidth=1.5)

    ax.set_xlabel("Time [ps]")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


if __name__ == "__main__":
    # --- Determine file to load ---
    # file_to_load = find_latest_result_file()
    file_to_load = "output/2025-04-12-17-30-15/current_1000uA.h5"
    print(f"Loading: {file_to_load}")

    sol = Solution.from_hdf5(file_to_load)

    # --- Paths for current extraction ---
    main_path = np.column_stack((np.linspace(-0.5, 1.5, 100), np.full(100, 2.5)))
    left_path = np.column_stack((np.linspace(-0.1, 0.1, 100), np.full(100, 0.3)))
    right_path = np.column_stack((np.linspace(2.7, 3.1, 100), np.full(100, 0.3)))

    # Create figure with 3 subplots arranged vertically
    fig = plt.figure(figsize=(12, 10), layout="constrained")

    # Create a grid with 3 rows: 2 for the original plots, 1 for the time series
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.3)

    # First subplot: 1x4 grid for positive current
    gs1 = gs[0].subgridspec(1, 4, width_ratios=[1, 1, 1, 1])
    axs_pos = [fig.add_subplot(gs1[i]) for i in range(4)]

    # Second subplot: 1x4 grid for negative current
    gs2 = gs[1].subgridspec(1, 4, width_ratios=[1, 1, 1, 1])
    axs_neg = [fig.add_subplot(gs2[i]) for i in range(4)]

    # Third subplot: time series plot
    ax_time = fig.add_subplot(gs[2])

    sc_vmin = 0
    sc_vmax = 3000

    # --- Supercurrent plots for positive current ---
    plot_supercurrent(axs_pos[0], sol, 50, vmin=sc_vmin, vmax=sc_vmax)
    plot_supercurrent(axs_pos[1], sol, 100, vmin=sc_vmin, vmax=sc_vmax)
    plot_supercurrent(axs_pos[2], sol, 200, colorbar=True, vmin=sc_vmin, vmax=sc_vmax)

    # --- Magnetic field for positive current ---
    x_pos = np.linspace(-0.7, 3.3, 50)
    y_pos = np.linspace(-3.75, 3.75, 50)
    field_pos = np.column_stack((np.tile(x_pos, 50), np.repeat(y_pos, 50)))

    imd = plot_magnetic_field(
        axs_pos[3], sol, field_pos, zs=0.01, solve_step=200, x_pos=x_pos, y_pos=y_pos
    )

    # --- Get current vs time for positive current ---
    times_pos, left_currents_pos, right_currents_pos = get_currents_vs_time(
        sol, left_path, right_path, dataset="supercurrent"
    )

    # --- Load negative current file ---
    file_to_load = "output/2025-04-12-17-30-15/current_-1000uA.h5"
    print(f"Loading: {file_to_load}")

    sol_neg = Solution.from_hdf5(file_to_load)

    # --- Supercurrent plots for negative current ---
    plot_supercurrent(axs_neg[0], sol_neg, 50, vmin=sc_vmin, vmax=sc_vmax)
    plot_supercurrent(axs_neg[1], sol_neg, 100, vmin=sc_vmin, vmax=sc_vmax)
    plot_supercurrent(
        axs_neg[2], sol_neg, 200, colorbar=True, vmin=sc_vmin, vmax=sc_vmax
    )

    # --- Magnetic field for negative current ---
    imh = plot_magnetic_field(
        axs_neg[3],
        sol_neg,
        field_pos,
        zs=0.01,
        solve_step=200,
        x_pos=x_pos,
        y_pos=y_pos,
    )

    # --- Get current vs time for negative current ---
    times_neg, left_currents_neg, right_currents_neg = get_currents_vs_time(
        sol_neg, left_path, right_path, dataset="supercurrent"
    )

    # --- Set titles and labels for spatial plots ---
    axs_pos[0].set_title("$t = 250~ps$ (+1000 μA)")
    axs_pos[1].set_title("$t = 500~ps$ (+1000 μA)")
    axs_pos[2].set_title("$t = 1000~ps$ (+1000 μA)")
    axs_pos[3].set_title("$B_z$ at $t = 1000~ps$ (+1000 μA)")

    axs_neg[0].set_title("$t = 250~ps$ (-1000 μA)")
    axs_neg[1].set_title("$t = 500~ps$ (-1000 μA)")
    axs_neg[2].set_title("$t = 1000~ps$ (-1000 μA)")
    axs_neg[3].set_title("$B_z$ at $t = 1000~ps$ (-1000 μA)")

    # Set equal aspect ratio for all spatial plots
    for ax in axs_pos + axs_neg:
        ax.set_aspect("equal")

    # Set labels
    for ax in axs_pos:
        ax.set_xlabel("x [μm]")
    for ax in axs_neg:
        ax.set_xlabel("x [μm]")

    axs_pos[0].set_ylabel("y [μm]")
    axs_neg[0].set_ylabel("y [μm]")

    # Add colorbars for magnetic field
    cbar_pos = fig.colorbar(
        imd, ax=axs_pos[3], label="$B_z$ [mT]", orientation="vertical"
    )
    cbar_neg = fig.colorbar(
        imh, ax=axs_neg[3], label="$B_z$ [mT]", orientation="vertical"
    )

    # --- Plot current time series ---
    # Combine positive and negative current data with time offset
    # Assume negative current simulation starts after positive one
    times_pos_array = np.array(times_pos)
    times_neg_array = np.array(times_neg)
    time_offset = times_pos_array[-1] + (
        times_pos_array[1] - times_pos_array[0]
    )  # Add one time step
    times_combined = np.concatenate([times_pos_array, times_neg_array + time_offset])
    left_combined = left_currents_pos + left_currents_neg
    right_combined = right_currents_pos + right_currents_neg

    plot_current_time_series(ax_time, times_combined, left_combined, right_combined)
    ax_time.set_title("Branch Currents vs Time")
    ax_time.axvline(
        x=time_offset, color="gray", linestyle="--", alpha=0.7, label="Current reversal"
    )
    ax_time.legend()

    plt.savefig(
        "output/2025-04-12-17-30-15/nmem_tdgl_simulation_with_time_series.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    # # --- Make animation ---
    output_path = os.path.dirname(file_to_load)
    tag = os.path.basename(file_to_load).split(".")[0]
    # video_html= make_animation_from_solution(
    #     sol_neg,
    #     output_path,
    #     tag,
    #     quantities=("supercurrent"),
    # )

    # output_path = os.path.dirname(file_to_load)
    # tag = os.path.basename(file_to_load).split(".")[0]

    # make_field_animation(sol_neg, output_path=output_path, tag=tag)

    fig, ax = plt.subplots(
        figsize=(5, 4),
        layout="constrained",
        height_ratios=[1],
        width_ratios=[1],
        sharex=True,
        sharey=True,
        subplot_kw={"aspect": "equal"},
    )
    # --- Supercurrent ---
    plot_supercurrent(ax, sol_neg, 200, colorbar=True, vmin=sc_vmin, vmax=sc_vmax)
    fig.patch.set_alpha(0.0)  # Set the figure background to transparent
    plt.savefig(
        "output/2025-04-12-17-30-15/nmem_tdgl_simulation_animation.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
