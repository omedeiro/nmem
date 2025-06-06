import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from nmem.analysis.plotting import darken, lighten, set_inter_font, set_pres_style


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
    **kwargs
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
        front_color = base_color
        top_color = lighten(base_color)
        side_color = darken(base_color)

        # Front face
        rect = Rectangle(
            (x - bar_width / 2, 0), bar_width, val, facecolor=front_color, edgecolor="none", **kwargs
        )
        ax.add_patch(rect)

        # Top face
        top = Polygon(
            [
                (x - bar_width / 2, val),
                (x - bar_width / 2 + depth / 1000, val + depth),
                (x + bar_width / 2 + depth / 1000, val + depth),
                (x + bar_width / 2, val),
            ],
            closed=True,
            facecolor=top_color,
            edgecolor="none",
            **kwargs
        )
        ax.add_patch(top)

        # Side face
        side = Polygon(
            [
                (x + bar_width / 2, 0),
                (x + bar_width / 2, val),
                (x + bar_width / 2 + depth / 1000, val + depth),
                (x + bar_width / 2 + depth / 1000, 0 + depth),
            ],
            closed=True,
            facecolor=side_color,
            edgecolor="none",
            **kwargs
        )
        ax.add_patch(side)

        # Text annotation
        ax.text(x, val + annotation_offset, f"{val} fJ", ha="center", va="bottom", fontsize=10)

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

        # Color shading
        front_color = base_color
        top_color = lighten(base_color, 1.1)
        side_color = darken(base_color, 0.6)

        # Front face
        rect = Rectangle(
            (0, y - bar_height / 2),
            val,
            bar_height,
            facecolor=front_color,
            edgecolor="none",
        )
        ax.add_patch(rect)

        # Top face
        top = Polygon(
            [
                (0, y + bar_height / 2),
                (depth, y + bar_height / 2 + depth),
                (val + depth, y + bar_height / 2 + depth),
                (val, y + bar_height / 2),
            ],
            closed=True,
            facecolor=top_color,
            edgecolor="none",
        )
        ax.add_patch(top)

        # Side face
        side = Polygon(
            [
                (val, y - bar_height / 2),
                (val, y + bar_height / 2),
                (val + depth, y + bar_height / 2 + depth),
                (val + depth, y - bar_height / 2 + depth),
            ],
            closed=True,
            facecolor=side_color,
            edgecolor="none",
        )
        ax.add_patch(side)

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

