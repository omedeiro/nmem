from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from nmem.analysis.bit_error import (
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_total_switches_norm,
)
from nmem.analysis.constants import (
    ALPHA,
    CRITICAL_TEMP,
    RETRAP,
    WIDTH,
)
from nmem.analysis.core_analysis import (
    get_fitting_points,
)
from nmem.analysis.currents import (
    calculate_branch_currents,
    calculate_state_currents,
    get_channel_temperature,
    get_critical_currents_from_trace,
    get_enable_current_sweep,
    get_enable_read_current,
    get_optimal_enable_read_current,
    get_optimal_enable_write_current,
    get_read_currents,
    get_state_currents_measured,
    get_write_current,
)
from nmem.analysis.styles import (
    CMAP,
    CMAP3,
)
from nmem.analysis.text_mapping import (
    get_text_from_bit,
)
from nmem.analysis.utils import (
    build_array,
    convert_cell_to_coordinates,
    filter_nan,
    get_current_cell,
)
from nmem.measurement.functions import (
    calculate_power,
)
from nmem.simulation.spice_circuits.plotting import (
    create_plot,
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


def plot_c2c3_comparison(ax, xfit, yfit, split_idx, label_c2="C2", label_c3="C3"):
    """
    Plot C2 and C3 comparison on a single axis.
    """
    ax.plot(xfit, yfit, label=label_c2, linestyle="-")
    plot_fitting(
        ax, xfit[split_idx + 1 :], yfit[split_idx + 1 :], label=label_c3, linestyle="-"
    )
    ax.set_ylim([0, 1000])
    ax.set_xlim([0, 500])
    ax.set_xlabel("Enable Current ($\mu$A)")
    ax.set_ylabel("Critical Current ($\mu$A)")
    ax.legend()
    return ax


def plot_c2c3_subplots(axs, xfit, yfit, split_idx, label_c2="C2", label_c3="C3"):
    """
    Plot C2 and C3 comparison on two subplots.
    """
    plot_fitting(
        axs[0],
        xfit[split_idx + 1 :],
        yfit[split_idx + 1 :],
        label=label_c3,
        linestyle="-",
    )
    axs[0].plot(xfit, yfit, label=label_c2, linestyle="-")
    axs[0].set_ylim([0, 1000])
    axs[0].set_xlim([0, 500])
    axs[0].set_xlabel("Enable Current ($\mu$A)")
    axs[0].set_ylabel("Critical Current ($\mu$A)")
    plot_fitting(
        axs[1], xfit[:split_idx], yfit[:split_idx], label=label_c3, linestyle="-"
    )
    axs[1].plot(xfit, yfit, label=label_c2, linestyle="-")
    axs[1].set_ylim([0, 1000])
    axs[1].set_xlim([0, 500])
    axs[1].set_xlabel("Enable Current ($\mu$A)")
    return axs


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



def plot_fill_between_array(ax: Axes, dict_list: list[dict]) -> Axes:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    for i, data_dict in enumerate(dict_list):
        plot_fill_between(ax, data_dict, colors[i])
    return ax


def plot_persistent_current(
    data_dict, persistent_current=75, critical_current_zero=1250
):
    """
    Plots calculated and measured persistent current curves for a given data_dict.
    Returns (fig, ax).
    """
    power = calculate_power(data_dict)
    fig, ax = plt.subplots()
    temperatures = np.linspace(0, CRITICAL_TEMP, 100)
    plot_calculated_state_currents(
        ax,
        temperatures,
        CRITICAL_TEMP,
        RETRAP,
        WIDTH,
        ALPHA,
        persistent_current,
        critical_current_zero,
    )
    plot_calculated_filled_region(
        ax,
        temperatures,
        data_dict,
        persistent_current,
        CRITICAL_TEMP,
        RETRAP,
        WIDTH,
        ALPHA,
        critical_current_zero,
    )
    return fig, ax


def plot_measured_state_currents(ax, mat_files, colors):
    """
    Plots measured state currents from a list of .mat files on the given axis.
    """
    for data_dict in mat_files:
        temp = data_dict["measured_temperature"].flatten()
        state_currents = data_dict["measured_state_currents"]
        for i in range(4):
            x, y = filter_nan(temp, state_currents[:, i])
            ax.plot(x, y, "-o", color=colors[i], label=f"State {i}")
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


def plot_cell_data(ax, data_dict, colors, markers):
    """
    Helper function to plot a single cell's data.
    """
    cell = get_current_cell(data_dict)
    column, row = convert_cell_to_coordinates(cell)
    x = data_dict["x"][0]
    y = data_dict["y"][0]
    ztotal = data_dict["ztotal"]
    xfit, yfit = get_fitting_points(x, y, ztotal)

    ax.plot(
        xfit,
        yfit,
        label=f"{cell}",
        color=colors[column],
        marker=markers[row],
        markeredgecolor="k",
        markeredgewidth=0.1,
    )
    ax.legend(loc="upper right", fontsize=6, labelspacing=0.1, handlelength=0.5)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 1500)

    return ax


def plot_grid(axs: Axes, dict_list: list[dict]) -> Axes:
    colors = CMAP3(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]

    for data_dict in dict_list:
        cell = get_current_cell(data_dict)
        column, row = convert_cell_to_coordinates(cell)
        ax = axs[row, column]
        ax = plot_cell_data(ax, data_dict, colors, markers)

        # xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.75)
        # plot_linear_fit(ax, xfit, yfit)

        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.xaxis.set_minor_locator(MultipleLocator(100))

    axs[-1, 0].set_xlabel("Enable Current [µA]")
    axs[-1, 0].set_ylabel("Critical Current [µA]")
    return axs


def plot_row(axs, dict_list):
    colors = CMAP3(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]

    for data_dict in dict_list:
        column, row = convert_cell_to_coordinates(get_current_cell(data_dict))
        ax = axs[column]
        ax = plot_cell_data(ax, data_dict, colors, markers)

    return axs


def plot_column(axs, dict_list):
    colors = CMAP3(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]

    for data_dict in dict_list:
        row = convert_cell_to_coordinates(get_current_cell(data_dict))[1]
        ax = axs[row]
        ax = plot_cell_data(ax, data_dict, colors, markers)

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


def plot_state_current_fit(ax, x_list, y_list, x_list_full, model, colors):
    for i in range(4):
        ax.plot(x_list[i], y_list[i], "-o", color=colors[i], label=f"State {i}")
        ax.plot(x_list_full, model[i], "--", color=colors[i])
    ax.legend()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [$\\mu$A]")
    ax.grid()
    ax.set_ybound(lower=0)
    ax.set_xbound(lower=0)





def plot_simulation_results(axs, ltsp_data_dict, case=16):
    """Plot simulation results for a given case on provided axes."""

    create_plot(axs, ltsp_data_dict, cases=[case])
    handles, labels = axs["T0"].get_legend_handles_labels()
    selected_labels = [
        "Left Branch Current",
        "Right Branch Current",
        "Left Critical Current",
        "Right Critical Current",
    ]
    selected_labels2 = [
        "$i_{\\mathrm{H_L}}$",
        "$i_{\\mathrm{H_R}}$",
        "$I_{\\mathrm{c,H_L}}$",
        "$I_{\\mathrm{c,H_R}}$",
    ]
    selected_handles = [handles[labels.index(lbl)] for lbl in selected_labels]
    return selected_handles, selected_labels2



