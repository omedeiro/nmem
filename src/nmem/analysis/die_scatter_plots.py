import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.plot_utils import apply_log_scale, set_axis_labels


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
