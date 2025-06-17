import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle

from nmem.analysis.styles import darken, lighten, set_inter_font, set_pres_style


def draw_extruded_bar_faces(
    ax, x, y, width, height, depth, orientation="v", base_color="#1f77b4", **kwargs
):
    """Draw 3D extruded bar on ax."""
    top_color = lighten(base_color)
    side_color = darken(base_color)

    if orientation == "v":
        # Vertical bar: front, top, side
        ax.add_patch(
            Rectangle(
                (x - width / 2, 0),
                width,
                height,
                facecolor=base_color,
                edgecolor="none",
                **kwargs,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (x - width / 2, height),
                    (x - width / 2 + depth / 1000, height + depth),
                    (x + width / 2 + depth / 1000, height + depth),
                    (x + width / 2, height),
                ],
                facecolor=top_color,
                edgecolor="none",
                **kwargs,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (x + width / 2, 0),
                    (x + width / 2, height),
                    (x + width / 2 + depth / 1000, height + depth),
                    (x + width / 2 + depth / 1000, 0 + depth),
                ],
                facecolor=side_color,
                edgecolor="none",
                **kwargs,
            )
        )
    elif orientation == "h":
        # Horizontal bar: front, top, side
        ax.add_patch(
            Rectangle(
                (0, y - width / 2),
                height,
                width,
                facecolor=base_color,
                edgecolor="none",
                **kwargs,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (0, y + width / 2),
                    (depth, y + width / 2 + depth),
                    (height + depth, y + width / 2 + depth),
                    (height, y + width / 2),
                ],
                facecolor=top_color,
                edgecolor="none",
                **kwargs,
            )
        )
        ax.add_patch(
            Polygon(
                [
                    (height, y - width / 2),
                    (height, y + width / 2),
                    (height + depth, y + width / 2 + depth),
                    (height + depth, y - width / 2 + depth),
                ],
                facecolor=side_color,
                edgecolor="none",
                **kwargs,
            )
        )


def plot_energy_extruded_bar(
    labels,
    energies_fj,
    colors,
    bar_width=0.6,
    depth=50,
    ax=None,
    xlabel="",
    ylabel="Energy per Operation [fJ]",
    title="Measured Energy of SNM Pulses",
    annotation_offset=80,
    ylim=(0, 1500),
    save_path=None,
    **kwargs,
):
    """
    Plot an extruded bar chart for energy per operation.

    Parameters
    ----------
    labels : list of str
        Bar labels.
    energies_fj : list of float
        Bar heights (energy values).
    colors : list of str
        Bar colors.
    bar_width : float
        Width of each bar.
    depth : float
        Depth for 3D effect.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates new.
    xlabel, ylabel, title : str
        Axis and plot labels.
    annotation_offset : float
        Offset for value annotation.
    ylim : tuple
        y-axis limits.
    save_path : str or None
        If given, saves the figure to this path.
    **kwargs : dict
        Passed to Rectangle and Polygon patches.
    """
    set_inter_font()
    set_pres_style()
    bar_positions = np.arange(len(labels))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    for i, (val, label, base_color) in enumerate(zip(energies_fj, labels, colors)):
        x = bar_positions[i]
        draw_extruded_bar_faces(
            ax,
            x,
            y=0,  # y is not used for vertical bars
            width=bar_width,
            height=val,
            depth=depth,
            orientation="v",
            base_color=base_color,
            **kwargs,
        )
        # Text annotation
        ax.text(
            x,
            val + annotation_offset,
            f"{val} fJ",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(*ylim)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, weight="bold")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    if save_path:
        fig.savefig(save_path, dpi=600)

    return ax


def draw_extruded_barh(
    ax, y_labels, values, colors, labels, xlabel, xticks, xticklabels
):
    bar_height = 0.6
    depth = 0.15

    for i, (val, label, base_color) in enumerate(zip(values, labels, colors)):
        y = i
        draw_extruded_bar_faces(
            ax,
            x=0,  # x is not used for horizontal bars
            y=y,
            width=bar_height,
            height=val,
            depth=depth,
            orientation="h",
            base_color=base_color,
        )
        # Label inside bar
        ax.text(
            val - 0.2,
            y,
            label,
            va="center",
            ha="right",
            fontsize=13,
            color="white" if base_color != "royalblue" else "black",
        )

    # Axes settings
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()  # larger bars on top
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    ax.set_xlim(0, max(values) + 1.5)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.grid(axis="x", linestyle="--", linewidth=0.5)


def plot_ber_3d_bar(ber_array: np.ndarray, total_trials: int = 200_000) -> Axes:
    barray = ber_array.copy()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

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
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("Errors")
    ax.set_title("Error Count per Cell")
    ax.set_xticks(np.arange(4) + 0.3)  # Offset tick locations
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_yticks(np.arange(4) + 0.3)  # Offset tick locations
    ax.set_yticklabels(["1", "2", "3", "4"])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_zlim(1, 500)
    ax.view_init(elev=45, azim=220)

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Errors (per 200k)")

    return ax


def plot_fidelity_clean_bar(ber_array: np.ndarray, total_trials: int = 200_000) -> Axes:
    import matplotlib.pyplot as plt
    import numpy as np

    # Prepare data
    barray = ber_array.copy().T
    ber_flat = barray.flatten()
    valid_mask = np.isfinite(ber_flat)
    ber_flat = ber_flat[valid_mask]
    fidelity_flat = 1 - ber_flat

    # Generate A1 to D4 labels
    rows = ["A", "B", "C", "D"]
    cols = range(1, 5)
    all_labels = [f"{r}{c}" for r in rows for c in cols]
    labels = np.array(all_labels)[valid_mask]

    # Error bars from binomial standard deviation
    errors = np.sqrt(ber_flat * (1 - ber_flat) / total_trials)
    # Clean label values
    display_values = [f"{f:.5f}" if f < 0.99999 else "≥0.99999" for f in fidelity_flat]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(len(fidelity_flat))
    ax.bar(x, fidelity_flat, yerr=errors, capsize=3, color="#658DDC", edgecolor="black")

    # Add value labels only if they fit in the visible range
    for i, (val, label) in enumerate(zip(fidelity_flat, display_values)):
        if val > 0.998:  # only plot label if within axis range
            ax.text(i, val + 1e-4, label, ha="center", va="bottom", fontsize=8)
        if val < 0.998:
            ax.text(i, 0.998 + 2e-4, label, ha="center", va="top", fontsize=8)
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
    ax.text(
        len(x) + 0.3,
        1 - 1.5e-3,
        "Previous Record",
        va="top",
        ha="right",
        fontsize=14,
        color="red",
        zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )
    ax.set_ylim(0.998, 1.0004)
    ax.set_yticks([0.998, 0.999, 0.9999])
    ax.set_yticklabels(["0.998", "0.999", "0.9999"])

    return ax


def plot_alignment_histogram(
    diff_list, binwidth=1, save_fig=False, output_path="alignment_histogram.pdf"
):
    """
    Plots a histogram of alignment differences.
    """
    n, bins, patches = plt.hist(
        x=diff_list,
        bins=range(
            int(np.floor(min(diff_list))),
            int(np.ceil(max(diff_list))) + binwidth,
            binwidth,
        ),
        edgecolor="black",
        color="#0504aa",
        alpha=0.5,
    )
    plt.ylabel("count")
    plt.xlabel("alignment difference [nm]")
    if save_fig:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    return n, bins, patches


def plot_alignment_offset_hist(
    dx_nm, dy_nm, save_fig=False, output_path="alignment_offsets_histogram.pdf"
):
    """
    Plots histograms of alignment offsets for ΔX and ΔY.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.hist(
        dx_nm, bins=20, edgecolor="black", color="#1f77b4", alpha=0.7, label="ΔX [nm]"
    )
    plt.hist(
        dy_nm, bins=20, edgecolor="black", color="#ff7f0e", alpha=0.7, label="ΔY [nm]"
    )
    plt.xlabel("Alignment Offset [nm]")
    plt.ylabel("Count")
    plt.title("Histogram of Alignment Offsets")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path, dpi=300)
    plt.show()
    return fig, ax


def plot_voltage_hist(ax: Axes, data_dict: dict) -> Axes:
    ax.hist(
        data_dict["read_zero_top"][0, :] * 1e3,
        log=True,
        range=(200, 600),
        bins=100,
        label="Read 0",
        color="#658DDC",
        alpha=0.8,
        zorder=-1,
    )
    ax.hist(
        data_dict["read_one_top"][0, :] * 1e3,
        log=True,
        range=(200, 600),
        bins=100,
        label="Read 1",
        color="#DF7E79",
        alpha=0.8,
    )
    ax.legend()
    return ax


def plot_alignment_stats(
    df_z,
    df_rot_valid,
    dx_nm,
    dy_nm,
    z_mean,
    z_std,
    r_mean,
    r_std,
    save=False,
    output_path="alignment_analysis.pdf",
):
    """
    Plots histograms and KDE for alignment statistics.
    """
    import seaborn as sns

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
    # Z height
    axs[0].hist(df_z["z_height_mm"], bins=20, edgecolor="black", color="#1f77b4")
    axs[0].set_xlabel("Z Height [mm]")
    axs[0].set_ylabel("Count")
    axs[0].text(
        0.97,
        0.97,
        f"$\\mu$ = {z_mean:.4f} mm\n$\\sigma$ = {z_std:.4f} mm",
        transform=axs[0].transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9
        ),
    )
    axs[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # Rotation
    axs[1].hist(
        df_rot_valid["rotation_mrad"], bins=20, edgecolor="black", color="#1f77b4"
    )
    axs[1].set_xlabel("Rotation [mrad]")
    axs[1].set_ylabel("Count")
    axs[1].text(
        0.97,
        0.97,
        f"$\\mu$ = {r_mean:.2f} mrad\n$\\sigma$ = {r_std:.2f} mrad",
        transform=axs[1].transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9
        ),
    )
    axs[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    # Alignment offsets
    ax = axs[2]
    sns.kdeplot(
        x=dx_nm,
        y=dy_nm,
        fill=True,
        cmap="crest",
        bw_adjust=0.7,
        levels=10,
        thresh=0.05,
        ax=ax,
    )
    ax.scatter(
        dx_nm,
        dy_nm,
        color="#333333",
        s=15,
        marker="o",
        label="Alignment Marks",
        alpha=0.8,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("ΔX [nm]")
    ax.set_ylabel("ΔY [nm]")
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(output_path, dpi=300)
    plt.show()
    return fig, axs


def plot_histogram(ax, vals, row_char, vmin=None, vmax=None):
    if len(vals) == 0:
        ax.text(
            0.5,
            0.5,
            f"No data\nfor row {row_char}",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.set_axis_off()
        return
    vals = vals[~np.isnan(vals)]
    log_bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 100)
    ax.hist(vals, bins=log_bins, color="#888", edgecolor="black", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim(10, 5000)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_ylabel(f"{row_char}", rotation=0, ha="right", va="center", fontsize=9)
    ax.tick_params(axis="both", which="both", labelsize=6)
    if vmin and vmax:
        ax.axvline(vmin, color="blue", linestyle="--", linewidth=1)
        ax.axvline(vmax, color="red", linestyle="--", linewidth=1)
