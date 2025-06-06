import os
from typing import List, Literal

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib import cm, ticker
from matplotlib import font_manager as fm
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm, Normalize, to_rgb
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from nmem.analysis.bit_error import (
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_total_switches_norm,
)
from nmem.analysis.core_analysis import (
    CRITICAL_TEMP,
    get_enable_write_width,
    get_fitting_points,
    get_read_width,
    get_voltage_trace_data,
    get_write_width,
    initialize_dict,
    process_cell,
)
from nmem.analysis.currents import (
    calculate_branch_currents,
    calculate_channel_temperature,
    calculate_state_currents,
    get_channel_temperature,
    get_channel_temperature_sweep,
    get_critical_current_heater_off,
    get_critical_currents_from_trace,
    get_enable_current_sweep,
    get_enable_read_current,
    get_enable_write_current,
    get_optimal_enable_read_current,
    get_optimal_enable_write_current,
    get_read_currents,
    get_state_current_markers,
    get_state_currents_measured,
    get_write_current,
)
from nmem.analysis.text_mapping import (
    get_text_from_bit,
)
from nmem.analysis.utils import (
    build_array,
    convert_cell_to_coordinates,
    create_rmeas_matrix,
    filter_plateau,
    get_current_cell,
)
from nmem.measurement.cells import CELLS

RBCOLORS = {0: "blue", 1: "blue", 2: "red", 3: "red"}
C0 = "#1b9e77"
C1 = "#d95f02"
CMAP = plt.get_cmap("coolwarm")
CMAP2 = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [C0, C1])
CMAP3 = plt.get_cmap("plasma").reversed()

READ_XMIN = 400
READ_XMAX = 1000
IC0_C3 = 910


def set_pres_style(dpi=600, font_size=14, grid_alpha=0.4):
    """
    Apply a presentation-optimized Matplotlib style.

    Parameters:
        dpi (int): Figure DPI (for saved files).
        font_size (int): Base font size for axes and labels.
        grid_alpha (float): Grid line transparency.
    """
    set_inter_font()
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "font.family": "Inter",
            "figure.figsize": (6, 4),
            "axes.titlesize": font_size + 4,
            "axes.labelsize": font_size + 2,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "font.size": font_size,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": grid_alpha,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
        }
    )

def darken(color, factor=0.6):
    return tuple(np.clip(factor * np.array(to_rgb(color)), 0, 1))


def lighten(color, factor=1.1):
    return tuple(np.clip(factor * np.array(to_rgb(color)), 0, 1))


def set_inter_font():
    if os.name == "nt":  # Windows
        font_path = r"C:\Users\ICE\AppData\Local\Microsoft\Windows\Fonts\Inter-VariableFont_opsz,wght.ttf"
    elif os.name == "posix":
        font_path = "/home/omedeiro/Inter-VariableFont_opsz,wght.ttf"
    else:
        font_path = None

    if font_path and os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        mpl.rcParams["font.family"] = "Inter"


def set_plot_style() -> None:
    set_inter_font()
    golden_ratio = (1 + 5**0.5) / 2  # ≈1.618
    width = 3.5  # Example width in inches (single-column for Nature)
    height = width / golden_ratio
    plt.rcParams.update(
        {
            "figure.figsize": [width, height],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.markersize": 3,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
        }
    )


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


def annotate_matrix(ax, R, fmt="{:.2g}", color="white"):
    """Add text annotations to matrix cells."""
    for y in range(R.shape[0]):
        for x in range(R.shape[1]):
            val = R[y, x]
            if not np.isnan(val):
                ax.text(x, y, fmt.format(val), ha="center", va="center", fontsize=6, color=color)


def get_log_norm_limits(R):
    """Safely get vmin and vmax for LogNorm."""
    values = R[~np.isnan(R) & (R > 0)]
    if values.size == 0:
        return None, None
    return np.nanmin(values), np.nanmax(values)



def plot_enable_sweep(
    ax: plt.Axes,
    dict_list: list[dict],
    range=None,
    add_errorbar=False,
    add_colorbar=False,
):
    N = len(dict_list)
    if range is not None:
        dict_list = dict_list[range]
    # ax, ax2 = plot_enable_write_sweep_multiple(ax, dict_list[0:6])
    ax = plot_enable_write_sweep_multiple(
        ax, dict_list, add_errorbar=add_errorbar, N=N, add_colorbar=add_colorbar
    )

    ax.set_ylabel("BER")
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    return ax


def plot_enable_write_sweep(ax: plt.Axes, dict_list: list[dict], **kwargs):
    colors = CMAP(np.linspace(0, 1, len(dict_list)))

    for j, data_dict in enumerate(dict_list):
        plot_read_sweep(
            ax,
            data_dict,
            "bit_error_rate",
            "enable_write_current",
            color=colors[j],
            **kwargs,
        )
        plot_fill_between(ax, data_dict, fill_color=colors[j])

    ax.set_xlabel("$I_{\mathrm{read}}$ [µA]")
    ax.set_ylabel("BER")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlim(READ_XMIN, READ_XMAX)
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    return ax


def plot_enable_write_temp(
    ax: plt.Axes, enable_write_currents, write_temperatures, colors=None
):
    colors = CMAP(np.linspace(0, 1, len(enable_write_currents)))
    ax.plot(
        enable_write_currents,
        write_temperatures,
        marker="o",
        color="black",
    )
    for i, idx in enumerate([0, 3, -6, -1]):
        ax.plot(
            enable_write_currents[idx],
            write_temperatures[idx],
            marker="o",
            markersize=6,
            markeredgecolor="black",
            markerfacecolor=colors[idx],
            markeredgewidth=0.2,
        )
    ax.set_xlabel("$I_{\mathrm{enable}}$ [µA]")
    ax.set_ylabel("$T_{\mathrm{write}}$ [K]")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    return ax


def plot_enable_read_sweep(ax: plt.Axes, dict_list, **kwargs):
    plot_read_sweep_array(
        ax, dict_list, "bit_error_rate", "enable_read_current", **kwargs
    )
    plot_fill_between_array(ax, dict_list)
    ax.axvline(IC0_C3, color="black", linestyle="--")
    ax.set_xlabel("$I_{\mathrm{read}}$ [µA]")
    ax.set_ylabel("BER")
    ax.set_xlim(READ_XMIN, READ_XMAX)
    return ax


def plot_enable_read_temp(ax: plt.Axes, enable_read_currents, read_temperatures):
    colors = CMAP(np.linspace(0, 1, len(enable_read_currents)))
    ax.plot(
        enable_read_currents,
        read_temperatures,
        marker="o",
        color="black",
        markersize=4,
    )
    enable_read_currents = enable_read_currents[::-1]
    read_temperatures = read_temperatures[::-1]
    for i in range(len(read_temperatures)):
        ax.plot(
            enable_read_currents[i],
            read_temperatures[i],
            marker="o",
            markersize=5,
            markeredgecolor="black",
            markerfacecolor=colors[i],
            markeredgewidth=0.2,
        )

    ax.set_xlabel("$I_{\mathrm{enable}}$ [µA]")
    ax.set_ylabel("$T_{\mathrm{read}}$ [K]")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

def plot_enable_sweep_markers(ax: plt.Axes, dict_list: list[dict]):
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.20))
    ax.set_ylim([8.3, 9.7])

    write_temp_array = np.empty((len(dict_list), 4))
    write_current_array = np.empty((len(dict_list), 1))
    enable_current_array = np.empty((len(dict_list), 4))
    for j, data_dict in enumerate(dict_list):
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_current = get_write_current(data_dict)
        write_temps = get_channel_temperature_sweep(data_dict)
        enable_currents = get_enable_current_sweep(data_dict)
        write_current_array[j] = write_current
        critical_current_zero = get_critical_current_heater_off(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                write_temp_array[j, i] = write_temps[arg]
                enable_current_array[j, i] = enable_currents[arg]
    markers = ["o", "s", "D", "^"]
    for i in range(4):
        ax.plot(
            enable_current_array[:, i],
            write_current_array,
            linestyle="--",
            marker=markers[i],
            markeredgecolor="k",
            markeredgewidth=0.5,
            color=RBCOLORS[i],
        )
    ax.set_ylim(0, 100)
    ax.set_xlim(250, 340)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.grid()
    ax.set_ylabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    ax.legend(
        [
            "$I_{1}$",
            "$I_{0}$",
            "$I_{0,\mathrm{inv}}$",
            "$I_{1,\mathrm{inv}}$",
        ],
        loc="upper right",
        frameon=True,
        ncol=2,
        facecolor="white",
        edgecolor="none",
    )



def add_colorbar(
    ax: plt.Axes,
    data_dict_list: list[dict],
    cbar_label: Literal["write_current", "enable_read_current"],
    cax=None,
):
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

def plot_write_sweep_formatted(ax: plt.Axes, dict_list: list[dict]):
    plot_write_sweep(ax, dict_list)
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(0, 300)
    return ax


def plot_write_sweep_formatted_markers(ax: plt.Axes, data_dict: dict):
    data = data_dict.get("data")
    data2 = data_dict.get("data2")
    colors = CMAP2(np.linspace(0, 1, 4))
    ax.plot(
        [d["write_current"] for d in data],
        [d["write_temp"] for d in data],
        "d",
        color=colors[0],
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    ax.plot(
        [d["write_current"] for d in data2],
        [d["write_temp"] for d in data2],
        "o",
        color=colors[2],
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$T_{\mathrm{write}}$ [K]")
    ax.set_xlim(0, 300)
    ax.legend(
        ["Lower bound", "Upper bound"],
        loc="upper right",
        fontsize=6,
        facecolor="white",
        frameon=True,
    )
    ax.grid()
    return ax


def plot_delay(ax: plt.Axes, data_dict: dict):
    delay_list = data_dict.get("delay")
    bit_error_rate = data_dict.get("bit_error_rate")
    N = 200e3
    sort_index = np.argsort(delay_list)
    delay_list = np.array(delay_list)[sort_index]
    bit_error_rate = np.array(bit_error_rate)[sort_index]
    bit_error_rate = np.array(bit_error_rate).flatten()
    ber_std = np.sqrt(bit_error_rate * (1 - bit_error_rate) / N)
    ax.errorbar(
        delay_list,
        bit_error_rate,
        yerr=ber_std,
        fmt="-",
        marker=".",
        color="black",
    )
    ax.set_ylabel("BER")
    ax.set_xlabel("Memory Retention Time (s)")

    ax.set_xscale("log")
    ax.set_xbound(lower=1e-6)
    ax.grid(True, which="both", linestyle="--")

    ax.set_yscale("log")
    ax.set_ylim([1e-4, 1e-3])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def plot_ber_grid(ax: plt.Axes):
    ARRAY_SIZE = (4, 4)
    param_dict = initialize_dict(ARRAY_SIZE)
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)

    plot_parameter_array(
        ax,
        xloc_list,
        yloc_list,
        param_dict["bit_error_rate"],
        log=True,
        cmap=plt.get_cmap("Blues").reversed(),
    )

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cax = ax.inset_axes([1.10, 0, 0.1, 1])
    fig = ax.get_figure()
    cbar = fig.colorbar(
        ax.get_children()[0], cax=cax, orientation="vertical", label="minimum BER"
    )

    return ax


def plot_bit_error_rate_args(ax: Axes, data_dict: dict, color) -> Axes:
    bit_error_rate = get_bit_error_rate(data_dict)
    berargs = get_bit_error_rate_args(bit_error_rate)

    read_current = get_read_currents(data_dict)
    for arg in berargs:
        if arg is not np.nan:
            ax.plot(
                read_current[arg],
                bit_error_rate[arg],
                marker="o",
                color=color,
            )
            ax.axvline(read_current[arg], color=color, linestyle="--")
    return ax


def plot_bit_error_rate(ax: Axes, data_dict: dict) -> Axes:
    bit_error_rate = get_bit_error_rate(data_dict)
    total_switches_norm = get_total_switches_norm(data_dict)
    measurement_param = data_dict.get("y")[0][:, 1] * 1e6

    ax.plot(
        measurement_param,
        bit_error_rate,
        color="#08519C",
        label="Bit Error Rate",
        marker=".",
    )
    ax.plot(
        measurement_param,
        total_switches_norm,
        color="grey",
        label="_Total Switches",
        linestyle="--",
        linewidth=1,
    )
    ax.legend()
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Normalized\nBit Error Rate")
    return ax


def plot_branch_currents(
    ax: Axes,
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    critical_current_zero: float,
) -> Axes:
    ichl, irhl, ichr, irhr = calculate_branch_currents(
        T, Tc, retrap_ratio, width_ratio, critical_current_zero
    )

    ax.plot(T, ichl, label="$I_{c, H_L}(T)$", color="b", linestyle="-")
    ax.plot(T, irhl, label="$I_{r, H_L}(T)$", color="b", linestyle="--")
    ax.plot(T, ichr, label="$I_{c, H_R}(T)$", color="r", linestyle="-")
    ax.plot(T, irhr, label="$I_{r, H_R}(T)$", color="r", linestyle="--")

    ax.plot(T, ichr + irhl, label="$I_{0}(T)$", color="g", linestyle="-")
    print(f"ichr: {ichr[0]}, irhl: {irhl[0]}")
    print(f"sum {ichr[0]+irhl[0]}")
    return ax


def plot_channel_temperature(
    ax: plt.Axes,
    data_dict: dict,
    channel_sweep: Literal["enable_write_current", "enable_read_current"],
    **kwargs,
) -> plt.Axes:
    if channel_sweep == "enable_write_current":
        temp = get_channel_temperature(data_dict, "write")
        current = get_enable_current_sweep(data_dict)
        print(f"temp: {temp}, current: {current}")
    else:
        temp = get_channel_temperature(data_dict, "read")
        current = get_enable_current_sweep(data_dict)
        print(f"temp: {temp}, current: {current}")
    ax.plot(current, temp, **kwargs)

    return ax


def plot_calculated_filled_region(
    ax,
    temp: np.ndarray,
    data_dict: dict,
    persistent_current: float,
    critical_temp: float,
    retrap_ratio: float,
    width_ratio: float,
    alpha: float,
    critical_current_zero: float,
) -> Axes:

    plot_calculated_nominal_region(
        ax,
        temp,
        data_dict,
        persistent_current,
        critical_temp,
        retrap_ratio,
        width_ratio,
        alpha,
        critical_current_zero,
    )
    plot_calculated_inverting_region(
        ax,
        temp,
        data_dict,
        persistent_current,
        critical_temp,
        retrap_ratio,
        width_ratio,
        alpha,
        critical_current_zero,
    )

    return ax


def plot_calculated_nominal_region(
    ax: Axes,
    temp: np.ndarray,
    data_dict: dict,
    persistent_current: float,
    critical_temp: float,
    retrap_ratio: float,
    width_ratio: float,
    alpha: float,
    critical_current_zero: float,
) -> Axes:
    i0, i1, i2, i3 = calculate_state_currents(
        temp,
        critical_temp,
        retrap_ratio,
        width_ratio,
        alpha,
        persistent_current,
        critical_current_zero,
    )

    upper_bound = np.minimum(i0, critical_current_zero)
    lower_bound = np.maximum(np.minimum(np.maximum(i3, i1), critical_current_zero), i2)
    ax.fill_between(
        temp,
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.1,
        hatch="////",
    )
    return ax


def plot_calculated_inverting_region(
    ax: Axes,
    temp: np.ndarray,
    data_dict: dict,
    persistent_current: float,
    critical_temp: float,
    retrap_ratio: float,
    width_ratio: float,
    alpha: float,
    critical_current_zero: float,
) -> Axes:
    i0, i1, i2, i3 = calculate_state_currents(
        temp,
        critical_temp,
        retrap_ratio,
        width_ratio,
        alpha,
        persistent_current,
        critical_current_zero,
    )
    upper_bound = np.minimum(np.minimum(np.maximum(i0, i2), i1), critical_current_zero)
    lower_bound = np.minimum(np.minimum(np.minimum(i2, i3), i0), critical_current_zero)
    ax.fill_between(
        temp,
        lower_bound,
        upper_bound,
        color="red",
        alpha=0.1,
        hatch="\\\\\\\\",
    )
    return ax


def plot_calculated_state_currents(
    ax: Axes,
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    alpha: float,
    persistent_current: float,
    critical_current_zero: float,
    **kwargs,
):
    i0, i1, i2, i3 = calculate_state_currents(
        T,
        Tc,
        retrap_ratio,
        width_ratio,
        alpha,
        persistent_current,
        critical_current_zero,
    )
    ax.plot(T, i0, label="$I_{{0}}(T)$", **kwargs)
    ax.plot(T, i1, label="$I_{{1}}(T)$", **kwargs)
    ax.plot(T, i2, label="$I_{{0,inv}}(T)$", **kwargs)
    ax.plot(T, i3, label="$I_{{1,inv}}(T)$", **kwargs)
    return ax


def plot_cell_parameter(ax: Axes, param: str) -> Axes:
    param_array = np.array([CELLS[cell][param] for cell in CELLS]).reshape(4, 4)
    plot_parameter_array(
        ax,
        np.arange(4),
        np.arange(4),
        param_array * 1e6,
        f"Cell {param}",
        log=False,
        norm=False,
        reverse=False,
    )
    return ax


def plot_critical_currents_from_trace(ax: Axes, dict_list: list) -> Axes:
    critical_currents, critical_currents_std = get_critical_currents_from_trace(
        dict_list
    )
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(dict_list)))
    heater_currents = [data["heater_current"].flatten() * 1e6 for data in dict_list]
    ax.errorbar(
        heater_currents,
        critical_currents,
        yerr=critical_currents_std,
        fmt="o",
        markersize=3,
        color=cmap[0, :],
    )
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("Heater Current [µA]")
    ax.set_ylabel("Critical Current [µA]")
    ax.set_ylim([0, 400])
    return ax


def plot_critical_currents_from_dc_sweep(
    ax: Axes, dict_list: list, substrate_temp: float = 1.3, save: bool = False
) -> Axes:

    critical_currents, critical_currents_std = get_critical_currents_from_trace(
        dict_list
    )

    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(dict_list)))
    heater_currents = np.array(
        [data["heater_current"].flatten() * 1e6 for data in dict_list]
    ).flatten()
    heater_currents = np.round(heater_currents, 0).astype(int)

    ax.plot(
        heater_currents,
        critical_currents,
        "o--",
        color=cmap[0],
        label="$I_{{EN}}$",
        linewidth=0.5,
        markersize=0.5,
        markerfacecolor=cmap[0],
    )

    ax.fill_between(
        heater_currents,
        critical_currents + critical_currents_std,
        critical_currents - critical_currents_std,
        color=cmap[0],
        alpha=0.3,
        edgecolor="none",
    )

    ax.set_xlabel("$I_{{EN}}$ [µA]")
    ax.set_ylabel("$I_{{c}}$ [µA]", labelpad=-1)
    ax.set_ylim([0, 400])
    ax.set_xlim([-500, 500])

    # Add secondary y-axis for temperature
    ax2 = ax.twinx()
    temp = calculate_channel_temperature(
        CRITICAL_TEMP, substrate_temp, np.abs(heater_currents), 500
    )
    ax2.plot(heater_currents, temp, "o--")
    ax2.set_ylim([0, 13])
    ax2.set_ylabel("Temperature [K]")
    ax2.axhline(substrate_temp, color="black", linestyle="--", linewidth=0.5)
    ax2.axhline(CRITICAL_TEMP, color="black", linestyle="--", linewidth=0.5)

    if save:
        plt.savefig("critical_currents_full.pdf", bbox_inches="tight")

    return ax


def plot_enable_current_relation(ax: Axes, dict_list: list[dict]) -> Axes:
    colors = CMAP(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]
    for data_dict in dict_list:
        cell = get_current_cell(data_dict)
        column, row = convert_cell_to_coordinates(cell)
        x, y, ztotal = build_array(data_dict, "total_switches_norm")
        xfit, yfit = get_fitting_points(x, y, ztotal)
        ax.plot(xfit, yfit, label=f"{cell}", color=colors[column], marker=markers[row])
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


def plot_fitting(ax: Axes, xfit: np.ndarray, yfit: np.ndarray, **kwargs) -> Axes:
    # xfit, yfit = filter_plateau(xfit, yfit, 0.98 * Ic0)
    ax.plot(xfit, yfit, **kwargs)
    plot_linear_fit(ax, xfit, yfit)

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


def plot_measured_state_current_list(ax: Axes, dict_list: list[dict]) -> Axes:
    sweep_length = len(dict_list)
    for j in range(0, sweep_length):
        plot_state_currents_measured(ax, dict_list[j], "enable_read_current")

    return ax


def plot_message(ax: Axes, message: str) -> Axes:
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(message):
        text = get_text_from_bit(bit, 3)
        ax.text(i + 0.5, axheight * 0.85, text, ha="center", va="center")

    return ax


def plot_parameter_array(
    ax: Axes,
    xloc: np.ndarray,
    yloc: np.ndarray,
    parameter_array: np.ndarray,
    title: str = None,
    log: bool = False,
    reverse: bool = False,
    cmap: plt.cm = None,
) -> Axes:
    if cmap is None:
        cmap = plt.get_cmap("viridis")
    if reverse:
        cmap = cmap.reversed()

    if log:
        ax.matshow(
            parameter_array,
            cmap=cmap,
            norm=LogNorm(vmin=np.min(parameter_array), vmax=np.max(parameter_array)),
        )
    else:
        ax.matshow(parameter_array, cmap=cmap)

    if title:
        ax.set_title(title)
    ax.set_xticks(range(4), ["A", "B", "C", "D"])
    ax.set_yticks(range(4), ["1", "2", "3", "4"])
    ax.tick_params(axis="both", length=0)
    return ax


def plot_enable_write_sweep_multiple(
    ax: Axes,
    dict_list: list[dict],
    add_errorbar: bool = False,
    N: int = None,
    add_colorbar: bool = True,
) -> Axes:
    if N is None:
        N = len(dict_list)
    colors = CMAP(np.linspace(0, 1, N))
    write_current_list = []
    for data_dict in dict_list:
        write_current = get_write_current(data_dict)
        write_current_list.append(write_current)

    for i, data_dict in enumerate(dict_list):
        write_current_norm = write_current_list[i] / 100
        plot_enable_sweep_single(
            ax, data_dict, color=CMAP(write_current_norm), add_errorbar=add_errorbar
        )

    if add_colorbar:
        norm = mcolors.Normalize(
            vmin=min(write_current_list), vmax=max(write_current_list)
        )  # Normalize for colormap
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.05, pad=0.05)
        cbar.set_label("Write Current [µA]")

    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    return ax


def plot_enable_sweep_single(
    ax: Axes,
    data_dict: dict,
    add_errorbar: bool = False,
    **kwargs,
) -> Axes:
    enable_currents = get_enable_current_sweep(data_dict)
    bit_error_rate = get_bit_error_rate(data_dict)
    write_current = get_write_current(data_dict)
    ax.plot(
        enable_currents,
        bit_error_rate,
        label=f"$I_{{W}}$ = {write_current:.1f}µA",
        **kwargs,
    )
    if add_errorbar:
        # ax.errorbar(
        #     enable_currents,
        #     bit_error_rate,
        #     yerr=np.sqrt(bit_error_rate * (1 - bit_error_rate) / len(bit_error_rate)),
        #     **kwargs,
        # )
        ax.fill_between(
            enable_currents,
            bit_error_rate
            - np.sqrt(bit_error_rate * (1 - bit_error_rate) / len(bit_error_rate)),
            bit_error_rate
            + np.sqrt(bit_error_rate * (1 - bit_error_rate) / len(bit_error_rate)),
            alpha=0.1,
            color=kwargs.get("color"),
        )
    ax.set_xlim(enable_currents[0], enable_currents[-1])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    return ax


def plot_read_sweep(
    ax: Axes,
    data_dict: dict,
    value_name: Literal["bit_error_rate", "write_0_read_1", "write_1_read_0"],
    variable_name: Literal[
        "enable_write_current",
        "read_width",
        "write_width",
        "write_current",
        "enable_read_current",
        "enable_write_width",
    ],
    add_errorbar: bool = False,
    **kwargs,
) -> Axes:
    write_temp = None
    label = None
    read_currents = get_read_currents(data_dict)

    if value_name == "bit_error_rate":
        value = get_bit_error_rate(data_dict)
    if value_name == "write_0_read_1":
        value = data_dict.get("write_0_read_1").flatten()
    if value_name == "write_1_read_0":
        value = data_dict.get("write_1_read_0").flatten()

    if variable_name == "write_current":
        variable = get_write_current(data_dict)
        label = f"{variable:.2f}µA"
    if variable_name == "enable_write_current":
        variable = get_enable_write_current(data_dict)
        write_temp = get_channel_temperature(data_dict, "write")
        if write_temp is None:
            label = f"{variable:.2f}µA"
        else:
            label = f"{variable:.2f}µA, {write_temp:.2f}K"
    if variable_name == "read_width":
        variable = get_read_width(data_dict)
        label = f"{variable:.2f} pts "
    if variable_name == "write_width":
        variable = get_write_width(data_dict)
        label = f"{variable:.2f} pts "
    if variable_name == "enable_read_current":
        variable = get_enable_read_current(data_dict)
        read_temp = get_channel_temperature(data_dict, "read")
        label = f"{variable:.2f}µA, {read_temp:.2f}K"
    if variable_name == "enable_write_width":
        variable = get_enable_write_width(data_dict)
        label = f"{variable:.2f} pts"

    ax.plot(
        read_currents,
        value,
        label=label,
        **kwargs,
    )
    if add_errorbar:
        # ax.errorbar(
        #     read_currents,
        #     value,
        #     yerr=np.sqrt(value * (1 - value) / len(value)),
        #     **kwargs,
        # )
        ax.fill_between(
            read_currents,
            value - np.sqrt(value * (1 - value) / len(value)),
            value + np.sqrt(value * (1 - value) / len(value)),
            alpha=0.1,
            color=kwargs.get("color"),
        )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    return ax


def plot_read_sweep_switch_probability(
    ax: Axes,
    data_dict: dict,
    **kwargs,
) -> Axes:
    read_currents = get_read_currents(data_dict)
    _, _, total_switch_probability = build_array(data_dict, "total_switches_norm")
    ax.plot(
        read_currents,
        total_switch_probability,
        **kwargs,
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylim(0, 1)
    return ax


def plot_fill_between_array(ax: Axes, dict_list: list[dict]) -> Axes:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    for i, data_dict in enumerate(dict_list):
        plot_fill_between(ax, data_dict, colors[i])
    return ax


def plot_read_sweep_array(
    ax: Axes,
    dict_list: list[dict],
    value_name: str,
    variable_name: str,
    colorbar=None,
    add_errorbar=False,
    **kwargs,
) -> Axes:
    colors = CMAP(np.linspace(0, 1, len(dict_list)))
    variable_list = []
    for i, data_dict in enumerate(dict_list):
        plot_read_sweep(
            ax,
            data_dict,
            value_name,
            variable_name,
            color=colors[i],
            add_errorbar=add_errorbar,
            **kwargs,
        )
        variable_list.append(data_dict[variable_name].flatten()[0] * 1e6)

    if colorbar is not None:
        norm = mcolors.Normalize(
            vmin=min(variable_list), vmax=max(variable_list)
        )  # Normalize for colormap
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.05, pad=0.05)
        cbar.set_label("Write Current [µA]")

    return ax

def plot_histogram(ax, vals, row_char, vmin=None, vmax=None):
    if len(vals) == 0:
        ax.text(0.5, 0.5, f"No data\nfor row {row_char}", ha="center", va="center", fontsize=8)
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
    ax.tick_params(axis='both', which='both', labelsize=6)
    if vmin and vmax:
        ax.axvline(vmin, color="blue", linestyle="--", linewidth=1)
        ax.axvline(vmax, color="red", linestyle="--", linewidth=1)

def plot_combined_histogram_and_die_maps(df, wafer_row_numbers, limit_dict, N=7):
    fig, axs = plt.subplots(
        len(wafer_row_numbers), N + 2,
        figsize=(5, 4),
        dpi=300,
        gridspec_kw={'width_ratios': [1] + [1]*N + [0.1]},
        constrained_layout=True
    )

    for i, row_number in enumerate(wafer_row_numbers):
        # Filter dies like A1, B1, ..., G1
        row_df = df[df["die"].str.endswith(str(row_number))].copy()
        valid_vals = row_df["Rmean"] / 1e3
        valid_vals = valid_vals[(valid_vals > 0) & np.isfinite(valid_vals) & (valid_vals < 50000)]

        n_nan = len(row_df) - len(valid_vals)
        if n_nan > 0:
            print(f"Row {row_number} has {n_nan} NaN values.")

        vmin, vmax = limit_dict.get(str(row_number), (valid_vals.min(), valid_vals.max()))
        plot_histogram(axs[i, 0], valid_vals, str(row_number), vmin, vmax)

        # Plot dies A1, B1, ..., G1
        im_list = []
        for j in range(N):
            die_name = f"{chr(65 + j)}{row_number}"
            die_df = df[df["die"] == die_name].copy()
            ax = axs[i, 1 + j]

            if die_df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=8)
                im_list.append(None)
                continue

            die_df["Rplot"] = die_df["Rmean"] / 1e3
            Rgrid = np.full((8, 8), np.nan)
            labels = np.full((8, 8), "", dtype=object)

            for _, row in die_df.iterrows():
                x, y = int(row["x_dev"]), int(row["y_dev"])
                if 0 <= x < 8 and 0 <= y < 8:
                    Rgrid[x, y] = row["Rplot"]
                    labels[x, y] = row["device"]

            im = ax.imshow(Rgrid.T, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax)
            im_list.append(im)

            # # Add device labels
            # for x in range(8):
            #     for y in range(8):
            #         label = labels[x, y]
            #         if label:
            #             ax.text(x, y, label, ha="center", va="center", fontsize=6, color="white")


            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(die_name, fontsize=7)
            ax.set_aspect("equal")

        # Colorbar
        axs[i, -1].set_xticks([])
        axs[i, -1].set_yticks([])
        axs[i, -1].set_frame_on(False)

        if any(im is not None for im in im_list):
            cax = axs[i, -1]
            first_valid_im = next(im for im in im_list if im is not None)
            cbar = fig.colorbar(first_valid_im, cax=cax)
            cbar.set_label("[kΩ]", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            cbar.set_ticks(np.linspace(vmin, vmax, 5))
            cbar.ax.set_yticklabels([f"{int(t)}" for t in np.linspace(vmin, vmax, 5)])
            if hasattr(cbar, "solids") and hasattr(cbar.solids, "set_rasterized"):
                cbar.solids.set_rasterized(True)
                cbar.solids.set_edgecolor("face")

    axs[-1, 0].set_xlabel("Resistance (kΩ)", fontsize=8)

    axs[2, 0].set_xlim(500, 1500)
    fig.patch.set_visible(False)
    save_fig = False
    if save_fig:
        fig.savefig("combined_wafer_map_and_histograms.pdf", bbox_inches="tight", dpi=300)
    plt.show()



def plot_read_switch_probability_array(
    ax: Axes, dict_list: list[dict], write_list=None, **kwargs
) -> Axes:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    print(f"len dict_list: {len(dict_list)}")
    for i, data_dict in enumerate(dict_list):
        if write_list is not None:
            plot_read_sweep_switch_probability(
                ax,
                data_dict,
                color=colors[i],
                label=f"{write_list[i]} µA",
                **kwargs,
            )
        else:
            plot_read_sweep_switch_probability(ax, data_dict, color=colors[i])
    return ax


def plot_read_delay(ax: Axes, dict_list: dict) -> Axes:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    for i, data_dict in enumerate(dict_list):
        read_currents = get_read_currents(data_dict)
        bit_error_rate = get_bit_error_rate(data_dict)
        ax.plot(
            read_currents,
            bit_error_rate,
            label=f"+{i+1}µs",
            color=colors[i],
            marker=".",
            markeredgecolor="k",
        )
    ax.set_xlim(read_currents[0], read_currents[-1])
    ax.set_yscale("log")
    return ax


def plot_write_sweep(ax: Axes, dict_list: str) -> Axes:
    # colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    write_temp_list = []
    enable_write_current_list = []
    for i, data_dict in enumerate(dict_list):
        x, y, ztotal = build_array(data_dict, "bit_error_rate")
        _, _, zswitch = build_array(data_dict, "total_switches_norm")
        write_temp = get_channel_temperature(data_dict, "write")
        enable_write_current = get_enable_write_current(data_dict)
        write_temp_list.append(write_temp)
        enable_write_current_list.append(enable_write_current)
        ax.plot(
            y,
            ztotal,
            label=f"$T_{{W}}$ = {write_temp:.2f} K, $I_{{EW}}$ = {enable_write_current:.2f} µA",
            color=colors[dict_list.index(data_dict)],
            marker=".",
        )
    ax.set_ylim([0, 1])
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    norm = mcolors.Normalize(
        vmin=min(enable_write_current_list), vmax=max(enable_write_current_list)
    )
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label("$I_{{EW}}$ [µA]")
    return ax


def plot_threshold(ax: Axes, start: int, end: int, threshold: float) -> Axes:
    ax.hlines(threshold, start, end, color="red", ls="-", lw=1)
    return ax


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


def plot_state_currents_measured_nominal(
    ax: Axes, nominal_read_temperature_list: list, nominal_state_currents_list: list
) -> Axes:
    for t, temp in enumerate(nominal_read_temperature_list):
        ax.plot(
            [temp, temp],
            nominal_state_currents_list[t],
            "o",
            linestyle="-",
            color="blue",
        )
    return ax


def plot_state_currents_measured_inverting(
    ax: Axes, inverting_read_temperature_list: list, inverting_state_currents_list: list
) -> Axes:
    for t, temp in enumerate(inverting_read_temperature_list):
        ax.plot(
            [temp, temp],
            inverting_state_currents_list[t],
            "o",
            linestyle="-",
            color="red",
        )
    return ax


def plot_state_currents_measured(ax: Axes, data_dict: dict, current_sweep: str) -> Axes:
    temp, state_currents = get_state_currents_measured(data_dict, current_sweep)

    if state_currents[0] is not np.nan:
        ax.plot(
            [temp, temp],
            state_currents[0:2],
            "o",
            linestyle="-",
            color="blue",
            label="_state0",
        )
    if state_currents[2] is not np.nan:
        ax.plot(
            [temp, temp],
            state_currents[2:4],
            "o",
            linestyle="-",
            color="red",
            label="_state1",
        )

    return ax


def plot_state_current_markers(
    ax: Axes,
    data_dict: dict,
    current_sweep: Literal["read_current", "enable_write_current"],
    **kwargs,
) -> Axes:

    state_current_markers = get_state_current_markers(data_dict, current_sweep)
    currents = state_current_markers[0, :]
    bit_error_rate = state_current_markers[1, :]
    if currents[0] > 0:
        for i in range(2):
            ax.plot(
                currents[i],
                bit_error_rate[i],
                color="blue",
                marker="o",
                markeredgecolor="k",
                linewidth=1.5,
                label="_state0",
                markersize=12,
                **kwargs,
            )
    if currents[2] > 0:
        for i in range(2, 4):
            ax.plot(
                currents[i],
                bit_error_rate[i],
                color="red",
                marker="o",
                markeredgecolor="k",
                linewidth=1.5,
                label="_state1",
                markersize=12,
                **kwargs,
            )
    return ax


def plot_voltage_trace(
    ax: Axes, time: np.ndarray, voltage: np.ndarray, **kwargs
) -> Axes:
    ax.plot(time, voltage, **kwargs)
    # ax.set_xlim(time[0], time[-1])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xticklabels([])
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="x", direction="in", which="both")
    # ax.grid(axis="x", which="both")
    return ax


def plot_voltage_trace_zoom(
    ax: Axes, x: np.ndarray, y: np.ndarray, start: float, end: float
) -> Axes:
    xzoom = x[(x > start) & (x < end)]
    yzoom = y[(x > start) & (x < end)]

    # smooth the yzoom data
    yzoom = np.convolve(yzoom, np.ones(20) / 20, mode="same")
    ax.plot(xzoom, 400 + yzoom * 10, color="red", ls="--", lw=1)
    ax.hlines(400, start, end, color="grey", ls="--", lw=1)

    return ax


def plot_voltage_trace_stack(
    axs: List[Axes], data_dict: dict, trace_index: int = 0
) -> List[Axes]:
    colors = CMAP(np.linspace(0.1, 1, 3))
    colors = np.flipud(colors)
    if len(axs) != 3:
        raise ValueError("The number of axes must be 3.")

    chan_in_x, chan_in_y = get_voltage_trace_data(
        data_dict, "trace_chan_in", trace_index
    )
    chan_out_x, chan_out_y = get_voltage_trace_data(
        data_dict, "trace_chan_out", trace_index
    )
    enab_in_x, enab_in_y = get_voltage_trace_data(data_dict, "trace_enab", trace_index)

    bitmsg_channel = data_dict.get("bitmsg_channel")[0]
    bitmsg_enable = data_dict.get("bitmsg_enable")[0]

    plot_voltage_trace(axs[0], chan_in_x, chan_in_y, color=colors[0], label="Input")

    if bitmsg_enable[1] == "W" and bitmsg_channel[1] != "N":
        plot_voltage_trace_zoom(axs[0], chan_in_x, chan_in_y, 0.9, 2.1)
        plot_voltage_trace_zoom(axs[0], chan_in_x, chan_in_y, 4.9, 6.1)

    if bitmsg_enable[3] == "W" and bitmsg_channel[3] != "N":
        plot_voltage_trace_zoom(axs[0], chan_in_x, chan_in_y, 2.9, 4.1)
        plot_voltage_trace_zoom(axs[0], chan_in_x, chan_in_y, 6.9, 8.1)

    plot_voltage_trace(axs[1], enab_in_x, enab_in_y, color=colors[1], label="Enable")

    plot_voltage_trace(axs[2], chan_out_x, chan_out_y, color=colors[2], label="Output")

    axs[2].xaxis.set_major_locator(MultipleLocator(5))
    axs[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    axs[2].set_xlim([0, 10])

    fig = plt.gcf()
    fig.supylabel("Voltage [mV]")
    fig.supxlabel("Time [µs]")
    fig.subplots_adjust(hspace=0.0)

    return axs


def plot_voltage_trace_averaged(
    ax: Axes, data_dict: dict, trace_name: str, **kwargs
) -> Axes:
    x, y = get_voltage_trace_data(data_dict, trace_name)
    ax.plot(
        (x - x[0]),
        y,
        **kwargs,
    )
    return ax


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
    # ax.set_xlabel("$V$ [mV]")
    # ax.set_ylabel("$N$")
    ax.legend()
    return ax


def plot_voltage_trace_bitstream(ax: Axes, data_dict: dict, trace_name: str) -> Axes:
    x, y = get_voltage_trace_data(
        data_dict,
        trace_name,
    )
    ax.plot(
        x,
        y,
        label=trace_name,
    )
    plot_message(ax, data_dict["bitmsg_channel"][0])
    return ax


def plot_current_voltage_curve(ax: Axes, data_dict: dict, **kwargs) -> Axes:
    time = data_dict.get("trace")[0, :]
    voltage = data_dict.get("trace")[1, :]

    M = int(np.round(len(voltage), -2))
    currentQuart = np.linspace(0, data_dict["vpp"] / 2 / 10e3, M // 4)
    current = np.concatenate(
        [-currentQuart, np.flip(-currentQuart), currentQuart, np.flip(currentQuart)]
    )

    if len(voltage) > M:
        voltage = voltage[:M]
        time = time[:M]
    else:
        voltage = np.concatenate([voltage, np.zeros(M - len(voltage))])
        time = np.concatenate([time, np.zeros(M - len(time))])

    ax.plot(voltage, current.flatten() * 1e6, **kwargs)
    return ax


def plot_current_voltage_from_dc_sweep(
    ax: Axes, dict_list: list, save: bool = False
) -> Axes:
    colors = plt.cm.coolwarm(np.linspace(0, 1, int(len(dict_list) / 2) + 1))
    colors = np.flipud(colors)
    for i, data in enumerate(dict_list):
        heater_current = np.abs(data["heater_current"].flatten()[0] * 1e6)
        ax = plot_current_voltage_curve(
            ax, data, color=colors[i], zorder=-i, label=f"{heater_current:.0f} µA"
        )
        if i == 10:
            break
    ax.set_ylim([-500, 500])
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [µA]")

    if save:
        plt.savefig("iv_curve.pdf", bbox_inches="tight")

    return ax


def plot_optimal_enable_currents(ax: Axes, data_dict: dict) -> Axes:
    cell = get_current_cell(data_dict)
    enable_read_current = get_optimal_enable_read_current(cell)
    enable_write_current = get_optimal_enable_write_current(cell)
    ax.vlines(
        [enable_write_current],
        *ax.get_ylim(),
        linestyle="--",
        color="grey",
        label="_Enable Write Current",
    )
    ax.vlines(
        [enable_read_current],
        *ax.get_ylim(),
        linestyle="--",
        color="r",
        label="_Enable Read Current",
    )
    return ax


def plot_grid(axs: Axes, dict_list: list[dict]) -> Axes:
    colors = CMAP3(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]
    for data_dict in dict_list:
        cell = get_current_cell(data_dict)

        column, row = convert_cell_to_coordinates(cell)
        x = data_dict["x"][0]
        y = data_dict["y"][0]
        ztotal = data_dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)
        axs[row, column].plot(
            xfit,
            yfit,
            label=f"{cell}",
            color=colors[column],
            marker=markers[row],
        )
        dd = sio.loadmat(
            "/home/omedeiro/nmem/src/nmem/analysis/enable_current_relation_v2/data2/SPG806_20250107_nMem_measure_enable_response_D6_A4_A3_2025-01-07 15-34-05.mat"
        )
        y_step_size = y[1] - y[0]
        # print(x)
        # print(y)
        # axs[row, column].errorbar(
        #     xfit,
        #     yfit,
        #     yerr=y_step_size * np.ones_like(yfit),
        #     fmt="o",
        #     color=colors[column],
        #     marker=markers[row],
        #     markeredgecolor="k",
        #     markeredgewidth=0.1,
        #     markersize=5,
        #     alpha=0.5,
        #     zorder=1,
        #     label="_data",
        # )

        xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.75)

        plot_linear_fit(
            axs[row, column],
            xfit,
            yfit,
        )
        # plot_optimal_enable_currents(axs[row, column], data_dict)
        axs[row, column].legend(loc="upper right")
        axs[row, column].xaxis.set_major_locator(MultipleLocator(500))
        axs[row, column].xaxis.set_minor_locator(MultipleLocator(100))
        axs[row, column].set_xlim(0, 600)
        axs[row, column].set_ylim(0, 1500)
    axs[-1, 0].set_xlabel("Enable Current [µA]")
    axs[-1, 0].set_ylabel("Critical Current [µA]")
    return axs


def plot_row(axs, dict_list):
    colors = CMAP3(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]
    for data_dict in dict_list:
        cell = get_current_cell(data_dict)
        column, row = convert_cell_to_coordinates(cell)
        x = data_dict["x"][0]
        y = data_dict["y"][0]
        ztotal = data_dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)

        axs[column].plot(
            xfit,
            yfit,
            label=f"{cell}",
            color=colors[column],
            marker=markers[row],
            markeredgecolor="k",
            markeredgewidth=0.1,
        )
        # plot_optimal_enable_currents(axs[column], data_dict)

        axs[column].legend(
            loc="upper right", fontsize=6, labelspacing=0.1, handlelength=0.5
        )
        axs[column].set_xlim(0, 600)
        axs[column].set_ylim(0, 1500)
    return axs


def plot_column(axs, dict_list):
    colors = CMAP3(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]
    for data_dict in dict_list:
        cell = get_current_cell(data_dict)
        column, row = convert_cell_to_coordinates(cell)
        x = data_dict["x"][0]
        y = data_dict["y"][0]
        ztotal = data_dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)

        axs[row].plot(
            xfit,
            yfit,
            label=f"{cell}",
            color=colors[column],
            marker=markers[row],
            markeredgecolor="k",
            markeredgewidth=0.1,
        )
        # plot_optimal_enable_currents(axs[row], data_dict)
        axs[row].legend(
            loc="upper right", fontsize=6, labelspacing=0.1, handlelength=0.5
        )
        axs[row].set_xlim(0, 600)
        axs[row].set_ylim(0, 1500)
    return axs


def plot_full_grid(axs, dict_list):
    plot_grid(axs[1:5, 0:4], dict_list)
    plot_row(axs[0, 0:4], dict_list)
    plot_column(axs[1:5, 4], dict_list)
    axs[0, 4].axis("off")
    axs[4, 0].set_xlabel("Enable Current [µA]")
    axs[4, 0].set_ylabel("Critical Current [µA]")
    return axs


def plot_waterfall(ax: Axes3D, dict_list: list[dict]) -> Axes3D:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    verts_list = []
    zlist = []
    for i, data_dict in enumerate(dict_list):
        enable_write_currents = get_enable_current_sweep(data_dict, "write")
        bit_error_rate = get_bit_error_rate(data_dict)
        write_current = get_write_current(data_dict)
        ax.plot(
            enable_write_currents,
            bit_error_rate,
            zs=write_current,
            zdir="y",
            color=colors[i],
            marker=".",
            markerfacecolor="k",
            markersize=5,
            linewidth=2,
        )
        zlist.append(write_current)
        verts = polygon_under_graph(enable_write_currents, bit_error_rate, 0.5)
        verts_list.append(verts)

    poly = PolyCollection(verts_list, facecolors=colors, alpha=0.6, edgecolors="k")
    ax.add_collection3d(poly, zs=zlist, zdir="y")

    ax.set_xlabel("$I_{{EW}}$ [µA]", labelpad=10)
    ax.set_ylabel("$I_W$ [µA]", labelpad=70)
    ax.set_zlabel("BER", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=12, pad=5)

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)

    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])
    ax.set_ylim(10, zlist[-1])
    ax.set_yticks(zlist)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_box_aspect([0.5, 1, 0.2], zoom=0.8)
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
