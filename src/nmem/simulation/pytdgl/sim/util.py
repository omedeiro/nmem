import os
from tdgl import SolverOptions, solve, Device, Solution
import numpy as np
import glob


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def time_dependent_current(t: float, current: float = 1000) -> float:
    """Return current in microamps at time t (in picoseconds)."""
    # Example: Gaussian pulse centered at t=1000ps with width 200ps
    return current * np.exp(-(((t - 500) / 200) ** 2))


def run_simulation(
    device: Device, current: float, stime: float, path: str, prev_sol: Solution = None
) -> Solution:
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, f"current_{int(current)}uA.h5")
    options = SolverOptions(
        skip_time=500,
        solve_time=stime,
        output_file=output_file,
        field_units="mT",
        current_units="uA",
        save_every=100,
    )
    return solve(
        device,
        options,
        terminal_currents=lambda t: {
            "source": time_dependent_current(t),
            "drain": -time_dependent_current(t),
        },
        applied_vector_potential=0,
        seed_solution=prev_sol,
    )


def find_latest_result_file(output_root: str = "output") -> str:
    """Finds the most recently modified .h5 file in the output directory tree."""
    files = glob.glob(f"{output_root}/**/*.h5", recursive=True)
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {output_root}")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def get_current_through_path(
    solution: Solution, path: np.ndarray, dataset="supercurrent", with_units=True
):
    if not isinstance(path, np.ndarray):
        path = np.array(path)
        if path.ndim != 2 or path.shape[1] != 2:
            raise ValueError("Path must be an Nx2 array of (x, y) coordinates.")

    return solution.current_through_path(path, with_units=with_units, dataset=dataset)


def make_animation_from_solution(
    solution: Solution,
    output_path: str,
    tag: str,
    quantities=("order_parameter", "phase"),
    fps: int = 20,
):
    from tdgl.visualization.animate import create_animation
    from IPython.display import HTML, display
    import h5py

    with h5py.File(solution.path, "r") as h5file:
        anim = create_animation(h5file, quantities=quantities, fps=fps)
        html = anim.to_html5_video()
    display(HTML(html))
    with open(os.path.join(output_path, f"{tag}.html"), "w") as f:
        f.write(html)


def make_field_animation(
    solution,
    output_path,
    tag="field",
    zs=0.01,
    vmin=-0.3,
    vmax=0.3,
    fps=20,
    resolution=50,
    dpi=100,
):
    """
    Create and save a magnetic field animation from a TDGL Solution object.

    Args:
        solution: tdgl.Solution object loaded from .h5
        output_path: where to save the mp4 file
        tag: filename prefix (no extension)
        zs: z-plane to evaluate field (μm)
        vmin, vmax: color scale limits
        fps: frames per second for video
        resolution: number of points per axis
        dpi: resolution of the saved video
    """
    print(f"Creating magnetic field animation with {solution.num_frames} frames...")

    # Define sampling grid
    x_pos = np.linspace(
        solution.mesh_bounds[0][0], solution.mesh_bounds[1][0], resolution
    )
    y_pos = np.linspace(
        solution.mesh_bounds[0][1], solution.mesh_bounds[1][1], resolution
    )
    field_pos = np.column_stack(
        (np.tile(x_pos, resolution), np.repeat(y_pos, resolution))
    )

    # Set up plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(
        field_pos[:, 0],
        field_pos[:, 1],
        c=np.zeros(len(field_pos)),
        s=10,
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Bₙ (mT)")
    ax.set_aspect("equal")
    ax.set_title("Magnetic Field $B_z$")
    ax.set_xlim(x_pos.min(), x_pos.max())
    ax.set_ylim(y_pos.min(), y_pos.max())

    # Frame update function
    def update(frame_idx):
        Bz = solution.get_field_at_positions(field_pos, zs=zs, frame=frame_idx)
        sc.set_array(Bz)
        ax.set_title(f"Magnetic Field $B_z$, Frame {frame_idx}")
        return (sc,)

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=solution.num_frames, interval=1000 / fps, blit=False
    )

    # Save to file
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, f"{tag}_field_animation.mp4")
    print(f"Saving animation to: {filename}")
    ani.save(filename, writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
