"""
This script plots the current transient through the left and right branches of the memory cell
as a function of time for both positive and negative applied currents.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from tdgl.solution.solution import Solution

from nmem.simulation.pytdgl.sim.util import (
    get_current_through_path,
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
    print(f"Processing {end_step - start_step + 1} time steps...")

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
    ax: plt.Axes,
    times: list,
    left_currents: list,
    right_currents: list,
    label_prefix: str = "",
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

    # Plot with different colors and line styles
    left_label = f"{label_prefix}Left branch" if label_prefix else "Left branch"
    right_label = f"{label_prefix}Right branch" if label_prefix else "Right branch"

    ax.plot(times, left_vals, label=left_label, linewidth=2.0)
    ax.plot(times, right_vals, label=right_label, linewidth=2.0)

    ax.set_xlabel("Time [ps]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    return ax, ylabel


if __name__ == "__main__":
    # --- Load simulation files ---
    file_pos = "output/2025-04-12-17-30-15/current_1000uA.h5"
    file_neg = "output/2025-04-12-17-30-15/current_-1000uA.h5"

    print(f"Loading positive current: {file_pos}")
    sol_pos = Solution.from_hdf5(file_pos)

    print(f"Loading negative current: {file_neg}")
    sol_neg = Solution.from_hdf5(file_neg)

    # --- Define current extraction paths ---
    # These paths define cut lines through the left and right branches
    left_path = np.column_stack((np.linspace(-0.1, 0.1, 100), np.full(100, 0.3)))
    right_path = np.column_stack((np.linspace(2.7, 3.1, 100), np.full(100, 0.3)))

    print("Extracting currents for positive applied current...")
    times_pos, left_currents_pos, right_currents_pos = get_currents_vs_time(
        sol_pos, left_path, right_path, dataset="supercurrent"
    )

    print("Extracting currents for negative applied current...")
    times_neg, left_currents_neg, right_currents_neg = get_currents_vs_time(
        sol_neg, left_path, right_path, dataset="supercurrent"
    )

    # --- Create the plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")

    # Plot positive current case
    ax1, ylabel = plot_current_time_series(
        ax1, times_pos, left_currents_pos, right_currents_pos, label_prefix=""
    )
    ax1.set_title("Branch Currents vs Time (+1000 μA Applied)")
    ax1.legend()

    # Plot negative current case
    ax2, _ = plot_current_time_series(
        ax2, times_neg, left_currents_neg, right_currents_neg, label_prefix=""
    )
    ax2.set_title("Branch Currents vs Time (-1000 μA Applied)")
    ax2.legend()

    # --- Save plots ---
    output_dir = "output/2025-04-12-17-30-15"
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(
        f"{output_dir}/branch_currents_separate.pdf",
        dpi=300,
        bbox_inches="tight",
    )


    print(f"Plot saved to {output_dir}/")
    plt.show()
