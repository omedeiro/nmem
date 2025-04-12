import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from tdgl import SolverOptions, solve, Device, Solution
from tqdm import tqdm
import imageio.v3 as iio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def time_dependent_current(t: float, current: float = 1000) -> float:
    """Return current in microamps at time t (in picoseconds)."""
    return current * np.exp(-(((t - 500) / 100) ** 2))


def run_simulation(
    device: Device, current: float, stime: float, path: str, prev_sol: Solution = None
) -> Solution:
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, f"current_{int(current)}uA.h5")
    options = SolverOptions(
        skip_time=50,
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
            "source": time_dependent_current(t, current),
            "drain": -time_dependent_current(t, current),
        },
        applied_vector_potential=0,
        seed_solution=prev_sol,
    )


def find_latest_result_file(output_root: str = "output") -> str:
    files = glob.glob(f"{output_root}/**/*.h5", recursive=True)
    if not files:
        raise FileNotFoundError(f"No .h5 files found under {output_root}")
    return max(files, key=os.path.getmtime)


def get_current_through_path(
    solution: Solution, path: np.ndarray, dataset="supercurrent", with_units=True
):
    path = np.asarray(path)
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

    with h5py.File(solution.path, "r") as h5file:
        anim = create_animation(
            h5file,
            quantities=quantities,
            fps=fps,
            output_file=os.path.join(output_path, f"{tag}.gif"),
        )
        html = anim.to_html5_video()

    display(HTML(html))


def make_field_animation(
    solution: Solution,
    output_path: str,
    tag="field",
    zs=0.01,
    vmin=-0.3,
    vmax=0.3,
    fps=20,
    resolution=150,
    dpi=100,
):
    """Create and save magnetic field animation with progress bar."""
    num_frames = solution.data_range[1] + 1
    print(f"Creating magnetic field animation ({num_frames} frames)...")

    # Sampling grid
    meshx = solution.device.points
    meshy = solution.device.points
    meshx = meshx[:, 0]
    meshy = meshy[:, 1]
    x_min, x_max = meshx.min(), meshx.max()
    y_min, y_max = meshy.min(), meshy.max()
    x_pos = np.linspace(x_min, x_max, resolution)
    y_pos = np.linspace(y_min, y_max, resolution)
    field_pos = np.column_stack(
        (np.tile(x_pos, resolution), np.repeat(y_pos, resolution))
    )
    # Plot setup
    fig, ax = plt.subplots(figsize=(5, 4))
    Bz0 = solution.field_at_position(field_pos, zs=zs).reshape(len(y_pos), len(x_pos))
    im = ax.imshow(
        Bz0,
        extent=[x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r",
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Bₙ (mT)")
    ax.set_title("Magnetic Field $B_z$")

    # Generate frames
    images = []
    for frame_idx in tqdm(range(num_frames), desc="Generating frames"):
        solution.solve_step = frame_idx
        Bz = solution.field_at_position(field_pos, zs=zs).reshape(len(y_pos), len(x_pos))
        im.set_data(Bz)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8").reshape(
            fig.canvas.get_width_height()[::-1] + (4,)
        )
        ax.set_title(f"Magnetic Field $B_z$, Frame {frame_idx}")

        images.append(image.copy())

    # Safety check
    if not images:
        raise RuntimeError("No frames were rendered — check that solve_step is working.")

    # Save animation
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, f"{tag}_field_animation.gif")
    print(f"Saving animation to: {filename}")
    iio.imwrite(filename, images, fps=fps)
    plt.close(fig)


def animate_currents(
    solution,
    output_path,
    tag="normal_current",
    dataset="normal_current",
    fps=20,
    resolution=100,
    dpi=100,
):
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, f"{tag}_stream_animation.gif")
    num_frames = solution.data_range[1] + 1

    x = solution.device.mesh.x
    y = solution.device.mesh.y
    X, Y = np.meshgrid(np.linspace(x.min(), x.max(), resolution),
                       np.linspace(y.min(), y.max(), resolution))
    
    fig, ax = plt.subplots(figsize=(5, 4))
    images = []

    for frame in tqdm(range(num_frames), desc="Animating streamplot"):
        solution.solve_step = frame
        Jx, Jy = solution.current_at_grid(X, Y, dataset=dataset)
        ax.clear()
        ax.streamplot(X, Y, Jx, Jy, density=1.2, linewidth=0.5, arrowsize=0.7)
        ax.set_title(f"{dataset} Streamplot - Frame {frame}")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_aspect("equal")

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        images.append(image.copy())

    print(f"Saving animation to: {filename}")
    iio.imwrite(filename, images, fps=fps)
    plt.close(fig)
