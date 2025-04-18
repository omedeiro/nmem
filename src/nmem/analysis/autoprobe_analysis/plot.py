import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from nmem.analysis.autoprobe_analysis.utils import (
    create_rmeas_matrix,
    get_log_norm_limits,
    annotate_matrix,
)
from matplotlib.colors import LogNorm


# Helper function to set axis labels and titles
def set_axis_labels(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)


# Helper function to apply log scale
def apply_log_scale(ax, logscale, axis="y"):
    if logscale:
        if axis == "y":
            ax.set_yscale("log")
        elif axis == "x":
            ax.set_xscale("log")




def plot_die_resistance_map(
    ax, df, die_name, cmap="turbo", logscale=True, annotate=False, vmin=None, vmax=None
):
    die_df = df[df["die"] == die_name]
    if die_df.empty:
        raise ValueError(f"No data found for die '{die_name}'")

    Rmeas = np.full((8, 8), np.nan)
    for _, row in die_df.iterrows():
        x, y = int(row["x_dev"]), int(row["y_dev"])
        y_dev = 7 - y  # Invert y-axis for display
        Rmeas[y_dev, x] = row["Rmean"] if row["Rmean"] > 0 else np.nan

    # Robust color limits using percentiles
    valid_vals = Rmeas[np.isfinite(Rmeas) & (Rmeas > 0)] / 1e3

    if valid_vals.size == 0:
        raise ValueError(f"Die {die_name} contains no valid (R > 0) data.")


    im = ax.imshow(
        Rmeas / 1e3,
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )

    if annotate:
        for y in range(8):
            for x in range(8):
                val = Rmeas[y, x]
                if np.isfinite(val):
                    ax.text(
                        x,
                        y,
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="white",
                    )

    ax.set_xticks([])
    ax.set_yticks([])
    return ax, im

    # ax.set_xticks(np.arange(8))
    # ax.set_yticks(np.arange(8))
    # ax.set_xticklabels(list("ABCDEFGH"))
    # ax.set_yticklabels(np.arange(1, 9))
    # set_axis_labels(
    #     ax, "Device Column", "Device Row", f"Resistance Map for Die {die_name}"
    # )
    # ax.set_aspect("equal")

    # if annotate:
    #     annotate_matrix(ax, Rmeas)

    # return ax, im


def plot_resistance_map(
    ax, df, grid_size=56, cmap="turbo", logscale=True, annotate=False
):
    Rmeas = create_rmeas_matrix(df, "x_abs", "y_abs", "Rmean", (grid_size, grid_size))
    if np.any(Rmeas == 0):
        Rmeas[Rmeas == 0] = np.nanmax(Rmeas)

    vmin, vmax = get_log_norm_limits(Rmeas)
    im = ax.imshow(
        Rmeas,
        origin="lower",
        extent=[0, grid_size, 0, grid_size],
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax) if logscale else None,
    )

    ax.set_xticks(np.linspace(3.5, 52.5, 7))
    ax.set_yticks(np.linspace(3.5, 52.5, 7))
    ax.set_xticklabels(list("ABCDEFG"))
    ax.set_yticklabels([str(i) for i in range(7, 0, -1)])
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.set_title("Autoprobe Resistance Map")

    for line in np.linspace(0, grid_size, 8):
        ax.axhline(line, color="k", lw=1.5)
        ax.axvline(line, color="k", lw=1.5)

    if annotate:
        annotate_matrix(ax, Rmeas.T)

    plt.colorbar(im, ax=ax, label="Resistance (Ω)")
    return ax


def plot_die_row(
    axes, df, row_number, cmap="turbo", annotate=False, vmin=None, vmax=None
):
    """
    Plot all dies in a given wafer row.
    row_number: 1 (top) to 7 (bottom)
    columns: 'A' (left) to 'G' (right)
    """
    if not (1 <= row_number <= 7):
        raise ValueError("row_number must be between 1 and 7")

    die_names = [f"{col}{row_number}" for col in "ABCDEFG"]
    im_list = []
    for ax, die_name in zip(axes, die_names):
        try:
            ax, im = plot_die_resistance_map(
                ax, df, die_name, cmap=cmap, annotate=annotate, vmin=vmin, vmax=vmax
            )
            im_list.append(im)
        except Exception as e:
            ax.set_title(f"{die_name} (Error)")
            ax.axis("off")
            print(f"Skipping {die_name}: {e}")

    return axes, im_list


def scatter_die_row_resistance(
    ax, df, row_number, marker="o", cmap="turbo", logscale=True
):
    """
    Plot a scatter of all Rmean values in a given wafer die row (top=1 to bottom=7).
    X-axis is the absolute device column position (0–55).
    """
    if not (1 <= row_number <= 7):
        raise ValueError("row_number must be between 1 and 7")

    colors = plt.get_cmap(cmap)

    for i, col_letter in enumerate("ABCDEFG"):
        die_name = f"{col_letter}{row_number}"
        die_df = df[df["die"] == die_name]

        if die_df.empty:
            continue

        # Compute global device x-position
        x_positions = die_df["x_abs"]
        resistances = die_df["Rmean"]

        ax.scatter(
            x_positions, resistances, label=die_name, marker=marker, color=colors(i / 6)
        )  # normalize i for colormap

    set_axis_labels(
        ax,
        "Absolute Device X Position",
        "Resistance (Ω)",
        f"Resistance Scatter Across Die Row {row_number}",
    )
    apply_log_scale(ax, logscale)
    ax.legend(title="Die")
    plot_quartile_lines(ax, resistances)

    return ax


def scatter_die_resistance(
    ax, df, die_name, marker="o", color="tab:blue", logscale=True
):
    """
    Plot a scatter of Rmean values for a single die.
    X-axis is the absolute device column position (x_abs).
    """
    die_df = df[df["die"] == die_name.upper()]
    if die_df.empty:
        raise ValueError(f"No data found for die '{die_name}'")

    x_positions = die_df["x_abs"]
    resistances = die_df["Rmean"]

    ax.scatter(
        x_positions, resistances, label=die_name.upper(), marker=marker, color=color
    )
    set_axis_labels(
        ax,
        "Absolute Device X Position",
        "Resistance (Ω)",
        f"Resistance Scatter for Die {die_name.upper()}",
    )
    apply_log_scale(ax, logscale)

    return ax


import numpy as np


def plot_quartile_lines(
    ax, data, color="gray", linestyle="--", linewidth=1.5, alpha=0.8
):
    """
    Compute and plot Q1 and Q3 horizontal lines on an existing axis.
    Adjust y-limits to [0.5 * Q1, 2 * Q3].

    Parameters:
    - ax: matplotlib axis to draw on
    - data: array-like, should be resistance values (Rmean)
    """
    data = np.array(data)
    data = data[np.isfinite(data) & (data > 0)]
    if len(data) == 0:
        return ax  # nothing to plot

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    ax.axhline(
        q1,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label="Q1 (25%)",
    )
    ax.axhline(
        q3,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label="Q3 (75%)",
    )

    ax.set_ylim(0.5 * q1, 2 * q3)
    return ax
