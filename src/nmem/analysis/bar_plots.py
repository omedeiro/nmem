import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

from nmem.analysis.utils import get_cell_labels
from nmem.analysis.styles import get_consistent_figure_size


def plot_ber_3d_bar(
    ber_array: np.ndarray,
    total_trials: int = 200_000,
    ax: Axes3D = None,
) -> tuple[plt.Figure, Axes3D]:
    """
    Plots a 3D bar chart for the Bit Error Rate (BER) array.

    Parameters:
    - ber_array: The array containing BER values.
    - total_trials: Total number of trials used for error count calculation.
    - ax: Optional pre-existing axis to plot on.
    - save_path: Path to save the plot as an image file. If None, the plot will be shown.

    Returns:
    - A tuple containing the figure and axis.
    """
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    barray = ber_array.copy()

    # 4x4 grid
    x_data, y_data = np.meshgrid(np.arange(4), np.arange(4))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    dz = barray.flatten()

    # Mask invalid entries
    valid_mask = np.isfinite(dz) & (dz < 5.5e-2)
    x_data, y_data, dz = x_data[valid_mask], y_data[valid_mask], dz[valid_mask]

    # Compute error counts
    error_counts = np.round(dz * total_trials).astype(int)
    z_data = np.zeros_like(error_counts)
    dx = dy = 0.6 * np.ones_like(error_counts)

    # Normalize error counts for colormap
    norm = Normalize(vmin=error_counts.min(), vmax=error_counts.max())
    colors = cm.Blues(norm(error_counts))

    # Plot 3D bars with color
    ax.bar3d(
        x_data,
        y_data,
        z_data,
        dx,
        dy,
        error_counts,
        shade=True,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    # Invert y-axis for logical row order
    ax.invert_yaxis()

    # Labels and ticks
    ax.set_xlabel("Column", labelpad=-8)
    ax.set_ylabel("Row", labelpad=-8)
    ax.set_zlabel("Errors", labelpad=-8)
    # ax.set_title("Error Count per Cell")
    ax.set_xticks(np.arange(4) + 0.3)  # Offset tick locations
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_yticks(np.arange(4) + 0.3)  # Offset tick locations
    ax.set_yticklabels(["1", "2", "3", "4"])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_zlim(1, 500)
    ax.view_init(elev=45, azim=220)


    ax.xaxis.set_tick_params(pad=-4)
    ax.yaxis.set_tick_params(pad=-4)
    ax.zaxis.set_tick_params(pad=-4)

    # Colorbar
    # mappable = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
    # mappable.set_array([])  # Empty array, just to set the color scale
    # cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    # cbar.set_label("Errors (per 200k)")

    return fig, ax


def plot_fidelity_clean_bar(
    ber_array: np.ndarray,
    total_trials: int = 200_000,
    ax: Axes = None,
) -> tuple[plt.Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=get_consistent_figure_size("wide"))
    else:
        fig = ax.figure

    # Prepare data
    barray = ber_array.copy().T
    ber_flat = barray.flatten()
    valid_mask = np.isfinite(ber_flat)
    ber_flat = ber_flat[valid_mask]
    fidelity_flat = 1 - ber_flat

    # Use centralized label function
    all_labels = get_cell_labels()
    labels = np.array(all_labels)[valid_mask]

    # Error bars from binomial standard deviation
    errors = np.sqrt(ber_flat * (1 - ber_flat) / total_trials)
    display_values = [f"{f:.5f}" if f < 0.99999 else "≥0.99999" for f in fidelity_flat]

    # Plotting
    x = np.arange(len(fidelity_flat))
    ax.bar(x, fidelity_flat, yerr=errors, capsize=3, color="#658DDC", edgecolor="black")

    # Add value labels only if they fit in the visible range
    # for i, (val, label) in enumerate(zip(fidelity_flat, display_values)):
    #     if val > 0.998:  # only plot label if within axis range
    #         ax.text(i, val + 1e-4, label, ha="center", va="bottom", fontsize=6)
    #     if val < 0.998:
    #         ax.text(i, 0.998 + 2e-4, label, ha="center", va="top", fontsize=6)
    # Formatting
    ax.set_xticks(x)
    ax.set_xlim(-1.5, len(x) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Fidelity (1 - BER)")
    ax.set_ylim(0.998, 1.0001)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Reference lines
    ax.axhline(0.999, color="#555555", linestyle="--", linewidth=0.8, zorder=3)
    ax.axhline(0.9999, color="#555555", linestyle="--", linewidth=0.8, zorder=3)
    ax.axhline(
        1 - 1.5e-3,
        color="red",
        linestyle="--",
        linewidth=0.8,
        zorder=3,
    )
    # ax.text(
    #     len(x) + 0.3,
    #     1 - 1.5e-3,
    #     "Previous Record",
    #     va="top",
    #     ha="right",
    #     fontsize=12,
    #     color="red",
    #     zorder=4,
    #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    # )
    ax.set_ylim(0.998, 1.0004)
    ax.set_yticks([0.998, 0.999, 0.9999])
    ax.set_yticklabels(["0.998", "0.999", "0.9999"])


    return fig, ax
