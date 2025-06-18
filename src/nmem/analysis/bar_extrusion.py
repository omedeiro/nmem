import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
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

    return ax

def plot_extruded_bar(
    labels,
    values,
    colors,
    orientation="v",
    bar_size=0.6,
    depth=50,
    ax=None,
    xlabel="",
    ylabel="",
    title="",
    bar_labels=None,
    annotation_offset=-0.15,
    axis_limits=None,
    save_path=None,
    xticks=None,
    xticklabels=None,
    fontsize=10,
    label_inside=True,
    **kwargs,
):
    """
    Plot extruded 3D-style bars in vertical or horizontal orientation.

    Parameters
    ----------
    labels : list of str
        Labels for each bar (y-axis in 'h', x-axis in 'v').
    values : list of float
        Bar heights (or widths in 'h').
    colors : list of str
        Bar face colors.
    orientation : str
        "v" for vertical bars, "h" for horizontal.
    bar_size : float
        Width (or height) of each bar.
    depth : float
        Depth for 3D shading effect.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.
    xlabel, ylabel, title : str
        Axis and plot labels.
    annotation_offset : float
        Offset for label text above/beside bars.
    axis_limits : tuple or None
        (ylim for 'v', xlim for 'h').
    save_path : str or None
        If provided, saves the figure.
    xticks, xticklabels : optional
        Tick positions and labels for horizontal plots.
    label_inside : bool
        Whether to draw labels inside horizontal bars.
    """
    set_inter_font()
    set_pres_style()

    positions = np.arange(len(labels))
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure


    if bar_labels is None:
        bar_labels = [f"{val:.2f}" for val in values]
        
    for i, (val, label, base_color) in enumerate(zip(values, bar_labels, colors)):
        x_or_y = positions[i]
        if orientation == "v":
            draw_extruded_bar_faces(
                ax,
                x=x_or_y,
                y=0,
                width=bar_size,
                height=val,
                depth=depth,
                orientation="v",
                base_color=base_color,
                **kwargs,
            )
            ax.text(
                x_or_y,
                val + val * annotation_offset,
                f"{val} fJ",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )
        else:
            draw_extruded_bar_faces(
                ax,
                x=0,
                y=x_or_y,
                width=bar_size,
                height=val,
                depth=depth,
                orientation="h",
                base_color=base_color,
                **kwargs,
            )
            label_color = "white" if base_color != "royalblue" else "black"
            ax.text(
                val + val * annotation_offset,
                x_or_y,
                label,
                va="center",
                ha="right" if label_inside else "left",
                fontsize=fontsize,
                color=label_color,
            )

    # Axis formatting
    if orientation == "v":
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(*axis_limits if axis_limits else (0, max(values) * 1.2))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.set_xlim(*axis_limits if axis_limits else (0, max(values) * 1.2))
        if xticks:
            ax.set_xticks(xticks)
        if xticklabels:
            ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5)

    if title:
        ax.set_title(title, weight="bold")

    if save_path:
        fig.savefig(save_path, dpi=600)

    return fig, ax

