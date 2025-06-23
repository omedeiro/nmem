import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_voltage_hist(ax: Axes, data_dict: dict) -> Axes:
    plot_general_histogram(
        data_dict["read_zero_top"][0, :] * 1e3,
        bins=100,
        color="#658DDC",
        alpha=0.8,
        label="Read 0",
        log=True,
        range=(200, 600),
        zorder=-1,
        ax=ax,
    )
    plot_general_histogram(
        data_dict["read_one_top"][0, :] * 1e3,
        bins=100,
        color="#DF7E79",
        alpha=0.8,
        label="Read 1",
        log=True,
        range=(200, 600),
        ax=ax,
    )
    ax.legend()
    return ax


def plot_histogram(ax, vals, row_char, vmin=None, vmax=None) -> Axes:
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
    plot_general_histogram(
        vals,
        bins=log_bins,
        color="#888",
        alpha=0.8,
        edgecolor="black",
        xscale="log",
        xlabel=None,
        ylabel=f"{row_char}",
        legend=False,
        grid=True,
        ax=ax,
    )
    ax.set_xlim(10, 5000)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", which="both", labelsize=6)
    if vmin and vmax:
        ax.axvline(vmin, color="blue", linestyle="--", linewidth=1)
        ax.axvline(vmax, color="red", linestyle="--", linewidth=1)

    return ax


def plot_general_histogram(
    data,
    bins=20,
    label=None,
    color="#1f77b4",
    alpha=0.7,
    edgecolor="black",
    xlabel=None,
    ylabel=None,
    title=None,
    log=False,
    xscale=None,
    yscale=None,
    range=None,
    legend=True,
    grid=True,
    ax=None,
    **kwargs,
) -> tuple[plt.Figure, Axes]:
    """
    Generalized histogram plotting function for consistent style and reuse.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    n, bins_out, patches = ax.hist(
        data,
        bins=bins,
        label=label,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        log=log,
        range=range,
        **kwargs,
    )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    if legend and label:
        ax.legend()
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    return fig, ax
