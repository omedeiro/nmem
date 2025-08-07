"""
Simple plotting script for branch currents from DC simulation results.
"""

import sys
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from tdgl.solution.solution import Solution

from nmem.simulation.pytdgl.sim.util import get_current_through_path


def plot_branch_currents(solution_path):
    """
    Simple plot of branch currents vs time.
    """
    print(f"Loading solution: {solution_path}")
    solution = Solution.from_hdf5(solution_path)

    # Get simulation info
    start_step, end_step = solution.data_range
    print(f"Time steps: {start_step} to {end_step}")

    # Define current extraction paths (cut lines through left and right branches)
    left_path = np.column_stack((np.linspace(-0.1, 0.1, 100), np.full(100, 0.3)))
    right_path = np.column_stack((np.linspace(2.7, 3.1, 100), np.full(100, 0.3)))

    # Extract currents vs time
    print("Extracting branch currents...")
    times = []
    left_currents = []
    right_currents = []

    # Sample every 5th step for speed
    for step in range(start_step, end_step + 1, 5):
        solution.solve_step = step

        # Get time (fallback to step number if time info not available)
        try:
            if hasattr(solution, "times") and solution.times is not None:
                time = solution.times[step]
            else:
                time = step
        except (IndexError, AttributeError):
            time = step

        times.append(time)

        # Extract currents through the cut lines
        left_current = get_current_through_path(
            solution, left_path, dataset="supercurrent", with_units=True
        )
        right_current = get_current_through_path(
            solution, right_path, dataset="supercurrent", with_units=True
        )

        left_currents.append(left_current)
        right_currents.append(right_current)

    # Convert to arrays
    times = np.array(times)

    # Handle units if present
    if len(left_currents) > 0 and hasattr(left_currents[0], "magnitude"):
        left_vals = np.array([curr.magnitude for curr in left_currents])
        right_vals = np.array([curr.magnitude for curr in right_currents])
        current_unit = str(left_currents[0].units)
    else:
        left_vals = np.array(left_currents)
        right_vals = np.array(right_currents)
        current_unit = "μA"

    # Create 2-panel plot
    fig = plt.figure(figsize=(16, 6))
    
    # Panel 1: Branch currents vs time
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(times, left_vals, label="Left branch", color="blue", linewidth=2)
    ax1.plot(times, right_vals, label="Right branch", color="red", linewidth=2)
    
    ax1.set_xlabel("Time [ps]")
    ax1.set_ylabel(f"Current [{current_unit}]")
    ax1.set_title("Branch Currents vs Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Supercurrent density with streamlines at final time
    print("Plotting supercurrent density...")
    solution.solve_step = end_step  # Set to final time step
    
    ax2 = plt.subplot(1, 2, 2)
    
    # Use TDGL's plotting function directly with our axis
    try:
        # Pass our axis directly to the TDGL plotting function
        fig_temp, ax_temp = solution.plot_currents(
            dataset="supercurrent", 
            streamplot=True, 
            colorbar=True,
            ax=ax2
        )
        
        print("✓ Supercurrent density plot created successfully")
        
    except Exception as e:
        print(f"Warning: Could not plot supercurrent density: {e}")
        ax2.text(0.5, 0.5, "Supercurrent plot\nfailed", 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

    # Plot the cut lines on the supercurrent plot
    ax2.plot(
        left_path[:, 0], left_path[:, 1], "b--", linewidth=3, label="Left cut line"
    )
    ax2.plot(
        right_path[:, 0], right_path[:, 1], "r--", linewidth=3, label="Right cut line"
    )

    ax2.set_xlabel("x (μm)")
    ax2.set_ylabel("y (μm)")
    ax2.set_title("Supercurrent Density with Cut Lines")
    ax2.set_aspect("equal")
    ax2.legend()

    plt.tight_layout()

    # Save plot
    output_dir = os.path.dirname(solution_path)
    base_name = os.path.splitext(os.path.basename(solution_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_branch_currents.pdf")

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {output_file}")

    # Print simple summary
    print(f"\nSummary:")
    print(f"  Left branch final:  {left_vals[-1]:.4f} {current_unit}")
    print(f"  Right branch final: {right_vals[-1]:.4f} {current_unit}")
    print(f"  Imbalance (L-R):    {left_vals[-1] - right_vals[-1]:.4f} {current_unit}")

    return fig, (ax1, ax2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_simple_currents.py <solution_file.h5>")
        sys.exit(1)

    solution_file = sys.argv[1]
    if not os.path.exists(solution_file):
        print(f"Error: File not found: {solution_file}")
        sys.exit(1)

    plot_branch_currents(solution_file)
