"""
DC current simulation script for applying constant 1 nA current to the memory cell.
This script applies a steady DC current instead of the time-dependent triangular waveform.
"""

import argparse
import os
from datetime import datetime
from tdgl import SolverOptions, solve

from nmem.simulation.pytdgl.devices.memory_cell import make_device
from nmem.simulation.pytdgl.sim.constants import XI


def run_dc_simulation(device, current_nA=1.0, stime=2000, path="output"):
    """
    Run simulation with constant DC current.

    Args:
        device: TDGL Device object
        current_nA: DC current in nanoamps
        stime: Simulation time in picoseconds
        path: Output directory path

    Returns:
        Solution object
    """
    os.makedirs(path, exist_ok=True)

    # Convert nA to μA for TDGL (which uses μA as default)
    current_uA = current_nA / 1000.0

    output_file = os.path.join(path, f"dc_current_{current_nA}nA.h5")

    options = SolverOptions(
        solve_time=stime,
        output_file=output_file,
        field_units="mT",
        current_units="uA",  # TDGL uses microamps
        save_every=50,  # Save more frequently for DC analysis
    )

    # DC current function - constant over time
    def dc_current_function(t):
        return {
            "source": current_uA,  # Positive current into source
            "drain": -current_uA,  # Negative current out of drain
        }

    print(
        f"Running DC simulation with {current_nA} nA ({current_uA} μA) for {stime} ps..."
    )

    solution = solve(
        device,
        options,
        terminal_currents=dc_current_function,
        applied_vector_potential=0,
    )

    return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DC current simulation for superconducting memory cell"
    )
    parser.add_argument(
        "--current", type=float, default=1.0, help="DC current in nanoamps"
    )
    parser.add_argument(
        "--time", type=float, default=2000, help="Simulation time in picoseconds"
    )
    parser.add_argument("--tag", type=str, default="", help="Tag for output folder")
    parser.add_argument(
        "--mesh_size", type=float, default=10.0, help="Mesh edge length factor (× ξ)"
    )

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args.tag:
        tagged_folder = f"dc_{args.tag}_{timestamp}"
    else:
        tagged_folder = f"dc_{timestamp}"

    output_path = os.path.join("output", tagged_folder)

    print("=" * 60)
    print("DC CURRENT SIMULATION")
    print("=" * 60)
    print(f"Current: {args.current} nA")
    print(f"Simulation time: {args.time} ps")
    print(f"Output directory: {output_path}")
    print("=" * 60)

    # Create device and mesh
    print("Creating device...")
    device = make_device()

    print(f"Creating mesh with max edge length: {args.mesh_size * XI:.6f} μm")
    device.make_mesh(max_edge_length=XI * args.mesh_size)

    print(f"Mesh statistics:")
    print(f"  - Number of points: {len(device.points)}")
    print(f"  - Number of triangles: {len(device.triangles)}")

    # Run simulation
    solution = run_dc_simulation(
        device, current_nA=args.current, stime=args.time, path=output_path
    )

    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {solution.path}")
    print(f"To analyze results:")
    print(f"  python plot_current_transient.py  # (if it exists)")
    print(f"  # or load in Python:")
    print(f"  from tdgl.solution.solution import Solution")
    print(f"  sol = Solution.from_hdf5('{solution.path}')")
    print("=" * 60)
