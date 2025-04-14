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
import matplotlib.font_manager as fm
import matplotlib as mpl


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


def find_latest_result_file(output_root: str = "output") -> str:
    """Finds the most recently modified .h5 file in the output directory tree."""
    files = glob.glob(f"{output_root}/**/*.h5", recursive=True)
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {output_root}")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


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
):
    sol.solve_step = solve_step
    
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

    fig, axs = plt.subplot_mosaic(
        """
        ABCD
        EFGH
        """,
        layout="constrained",
        height_ratios=[1, 1],
        width_ratios=[1, 1, 1, 1],
        sharex=True,
        sharey=True,
        subplot_kw={"aspect": "equal"},
    )

    sc_vmin = 0
    sc_vmax = 3000

    # --- Supercurrent ---
    plot_supercurrent(axs["A"], sol, 50, vmin=sc_vmin, vmax=sc_vmax)
    plot_supercurrent(axs["B"], sol, 100, vmin=sc_vmin, vmax=sc_vmax)
    plot_supercurrent(axs["C"], sol, 200, colorbar=True, vmin=sc_vmin, vmax=sc_vmax)

    # --- Magnetic field ---
    x_pos = np.linspace(-0.7, 3.3, 50)
    y_pos = np.linspace(-3.75, 3.75, 50)
    field_pos = np.column_stack((np.tile(x_pos, 50), np.repeat(y_pos, 50)))

    imd = plot_magnetic_field(axs["D"], sol, field_pos, zs=0.01, solve_step=200)

    file_to_load = "output/2025-04-12-17-30-15/current_-1000uA.h5"
    print(f"Loading: {file_to_load}")

    sol = Solution.from_hdf5(file_to_load)
    plot_supercurrent(axs["E"], sol, 50,  vmin=sc_vmin, vmax=sc_vmax)
    plot_supercurrent(axs["F"], sol, 100, vmin=sc_vmin, vmax=sc_vmax)
    plot_supercurrent(axs["G"], sol, 200, colorbar=True, vmin=sc_vmin, vmax=sc_vmax)

    imh = plot_magnetic_field(axs["H"], sol, field_pos, zs=0.01, solve_step=200)

    axs["A"].set_title("$t = 250~ps$")
    axs["B"].set_title("$t = 500~ps$")
    axs["C"].set_title("$t = 1000~ps$")
    axs["D"].set_title("$t = 1000~ps$")
    axs["A"].set_xlabel("x [μm]")
    axs["B"].set_xlabel("x [μm]")
    axs["C"].set_xlabel("x [μm]")
    axs["D"].set_xlabel("x [μm]")
    cbar = fig.colorbar(
        imd, ax=axs["D"], label="$B_z$ [mT]", orientation="vertical",
    )
    axs["A"].set_ylabel("y [μm]")
    axs["E"].set_ylabel("y [μm]")
    axs["B"].set_ylabel(None)
    axs["C"].set_ylabel(None)

    axs["E"].set_title("$t = 250~ps$")
    axs["F"].set_title("$t = 500~ps$")
    axs["G"].set_title("$t = 1000~ps$")
    axs["H"].set_title("$t = 1000~ps$")
    axs["E"].set_xlabel("x [μm]")
    axs["F"].set_xlabel("x [μm]")
    axs["G"].set_xlabel("x [μm]")
    axs["H"].set_xlabel("x [μm]")
    
    axs["F"].set_ylabel(None)
    axs["G"].set_ylabel(None)
    cbar = fig.colorbar(
        imh, ax=axs["H"], label="$B_z$ [mT]", orientation="vertical",
    )
    

    plt.savefig(
        "output/2025-04-12-17-30-15/nmem_tdgl_simulation.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    # # --- Make animation ---
    output_path = os.path.dirname(file_to_load)
    tag = os.path.basename(file_to_load).split(".")[0]
    # video_html= make_animation_from_solution(
    #     sol,
    #     output_path,
    #     tag,
    #     quantities=("supercurrent"),
    # )

    # output_path = os.path.dirname(file_to_load)
    # tag = os.path.basename(file_to_load).split(".")[0]

    # make_field_animation(sol, output_path=output_path, tag=tag)

