from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection

from nmem.analysis.bit_error import get_bit_error_rate
from nmem.analysis.currents import get_enable_current_sweep, get_enable_read_current
from nmem.analysis.styles import CMAP
from nmem.analysis.text_mapping import (
    get_text_from_bit,
)


def plot_message(ax: Axes, message: str) -> Axes:
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(message):
        text = get_text_from_bit(bit, 3)
        ax.text(i + 0.5, axheight * 0.85, text, ha="center", va="center")

    return ax



def plot_fill_between_array(ax: Axes, dict_list: list[dict]) -> Axes:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    for i, data_dict in enumerate(dict_list):
        plot_fill_between(ax, data_dict, colors[i])
    return ax



def plot_fitting(ax: Axes, xfit: np.ndarray, yfit: np.ndarray, add_text:bool=False, **kwargs) -> Axes:
    # xfit, yfit = filter_plateau(xfit, yfit, 0.98 * Ic0)
    ax.plot(xfit, yfit, **kwargs)
    plot_linear_fit(ax, xfit, yfit, add_text=add_text)

    return ax



def plot_linear_fit(
    ax: Axes, xfit: np.ndarray, yfit: np.ndarray, add_text: bool = False
) -> Axes:
    z = np.polyfit(xfit, yfit, 1)
    p = np.poly1d(z)
    x_intercept = -z[1] / z[0]
    # ax.scatter(xfit, yfit, color="#08519C")
    xplot = np.linspace(0, x_intercept, 10)
    ax.plot(xplot, p(xplot), ":", color="k")
    if add_text:
        ax.text(
            0.1,
            0.1,
            f"{p[1]:.3f}x + {p[0]:.3f}\n$x_{{int}}$ = {x_intercept:.2f}",
            fontsize=12,
            color="red",
            backgroundcolor="white",
            transform=ax.transAxes,
        )
    ax.set_xlim((0, 600))
    ax.set_ylim((0, 2000))

    return ax


def plot_fill_between(ax, data_dict, fill_color):
    # fill the area between 0.5 and the curve
    enable_write_currents = get_enable_current_sweep(data_dict)
    bit_error_rate = get_bit_error_rate(data_dict)
    verts = polygon_nominal(enable_write_currents, bit_error_rate)
    poly = PolyCollection([verts], facecolors=fill_color, alpha=0.3, edgecolors="k")
    ax.add_collection(poly)
    verts = polygon_inverting(enable_write_currents, bit_error_rate)
    poly = PolyCollection([verts], facecolors=fill_color, alpha=0.3, edgecolors="k")
    ax.add_collection(poly)

    return ax


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



def polygon_under_graph(x, y, y2=0.0):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], y2), *zip(x, y), (x[-1], y2)]


def polygon_nominal(x: np.ndarray, y: np.ndarray) -> list:
    y = np.copy(y)
    y[y > 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def polygon_inverting(x: np.ndarray, y: np.ndarray) -> list:
    y = np.copy(y)
    y[y < 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def get_log_norm_limits(R):
    """Safely get vmin and vmax for LogNorm."""
    values = R[~np.isnan(R) & (R > 0)]
    if values.size == 0:
        return None, None
    return np.nanmin(values), np.nanmax(values)





def add_colorbar(
    ax: plt.Axes,
    data_dict_list: list[dict],
    cbar_label: Literal["write_current", "enable_read_current"],
    cax=None,
)-> plt.colorbar:
    data_list = []
    for data_dict in data_dict_list:
        if cbar_label == "write_current":
            data_list += [d["write_current"] * 1e6 for d in data_dict]
            label = "Write Current [µA]"
        elif cbar_label == "enable_read_current":
            enable_read_current = [get_enable_read_current(d) for d in data_dict]
            # print(f"Enable Read Current: {enable_read_current}")
            # data_list += [enable_read_current]
            data_list = enable_read_current
            label = "$I_{{ER}}$ [µA]"

    norm = mcolors.Normalize(vmin=min(data_list), vmax=max(data_list))
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])

    if cax is not None:
        cbar = plt.colorbar(sm, cax=cax)
    else:
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.05, pad=0.05)

    cbar.set_label(label)
    return cbar



def plot_text_labels(
    ax: Axes, xloc: np.ndarray, yloc: np.ndarray, ztotal: np.ndarray, log: bool
) -> Axes:
    for x, y in zip(xloc, yloc):
        text = f"{ztotal[y, x]:.2f}"
        txt_color = "black"
        if ztotal[y, x] > (0.8 * max(ztotal.flatten())):
            txt_color = "white"
        if log:
            text = f"{ztotal[y, x]:.1e}"
            txt_color = "black"

        ax.text(
            x,
            y,
            text,
            color=txt_color,
            backgroundcolor="none",
            ha="center",
            va="center",
            weight="bold",
        )

    return ax

def plot_threshold(ax: Axes, start: int, end: int, threshold: float) -> Axes:
    ax.hlines(threshold, start, end, color="red", ls="-", lw=1)
    return ax


