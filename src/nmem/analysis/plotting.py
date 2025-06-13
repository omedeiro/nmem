from typing import List, Literal

import ltspice
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from nmem.analysis.bit_error import (
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_total_switches_norm,
)
from nmem.analysis.constants import (
    ALPHA,
    CRITICAL_CURRENT_ZERO,
    CRITICAL_TEMP,
    IC0_C3,
    IRHL_TR,
    IRM,
    PROBE_STATION_TEMP,
    READ_XMAX,
    READ_XMIN,
    RETRAP,
    SUBSTRATE_TEMP,
    WIDTH,
)
from nmem.analysis.core_analysis import (
    get_enable_write_width,
    get_fitting_points,
    get_read_width,
    get_voltage_trace_data,
    get_write_width,
)
from nmem.analysis.currents import (
    calculate_branch_currents,
    calculate_channel_temperature,
    calculate_critical_current_temp,
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
    filter_nan,
    filter_plateau,
    get_current_cell,
)
from nmem.measurement.cells import (
    CELLS,
)
from nmem.measurement.functions import (
    calculate_power,
)
from nmem.simulation.spice_circuits.functions import process_read_data
from nmem.simulation.spice_circuits.plotting import (
    create_plot,
    plot_current_sweep_ber,
    plot_current_sweep_switching,
)

from nmem.analysis.styles import (
    CMAP,
    CMAP2,
    CMAP3,
    RBCOLORS,
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


def plot_write_current_sweep(
    ax: plt.Axes, dict_list: list[dict[str, list[float]]]
) -> plt.Axes:
    plot_read_sweep_array(
        ax, dict_list, "bit_error_rate", "write_current", add_errorbar=False
    )
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    # ax.legend(
    #     frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left", title="Write Current"
    # )

    return ax


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


def plot_enable_write_sweep2(
    dict_list, save_fig=False, output_path="enable_write_sweep.pdf"
):
    """
    Plots the enable write sweep for multiple datasets.
    Returns (fig, ax).
    """
    fig, ax = plt.subplots()
    ax = plot_enable_write_sweep_multiple(ax, dict_list)
    ax.set_xlabel("$I_{\\mathrm{enable}}$ [$\\mu$A]")
    ax.set_ylabel("BER")
    if save_fig:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def plot_state_current_markers2(dict_list):
    """
    Plots state current markers for each dataset.
    Returns (fig, ax).
    """
    colors = {
        0: "red",
        1: "red",
        2: "blue",
        3: "blue",
    }
    fig, ax = plt.subplots()
    for data_dict in dict_list:
        state_current_markers = get_state_current_markers(
            data_dict, "enable_write_current"
        )
        write_current = get_write_current(data_dict)
        for i, state_current in enumerate(state_current_markers[0, :]):
            if state_current > 0:
                ax.plot(
                    write_current,
                    state_current,
                    "o",
                    label=f"{write_current} $\\mu$A",
                    markerfacecolor=colors[i],
                    markeredgecolor="none",
                )
    ax.set_xlabel("$I_{\\mathrm{write}}$ [$\\mu$A]")
    ax.set_ylabel("$I_{\\mathrm{enable}}$ [$\\mu$A]")
    return fig, ax


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


def plot_write_current_enable_sweep_margin(
    dict_list,
    inner,
    save_fig=False,
    output_path="write_current_enable_sweep_margin.pdf",
):
    """
    Plots the write current enable sweep margin using a subplot mosaic.
    Returns (fig, axs).
    """
    fig, axs = plt.subplot_mosaic(inner, figsize=(6, 2))
    sort_dict_list = sorted(
        dict_list, key=lambda x: x.get("write_current").flatten()[0]
    )
    ax = axs["A"]
    plot_enable_sweep(
        ax, sort_dict_list, range=slice(0, len(sort_dict_list), 2), add_colorbar=True
    )
    ax = axs["B"]
    plot_enable_sweep_markers(ax, sort_dict_list)
    fig.subplots_adjust(wspace=0.7, hspace=0.5)
    if save_fig:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, axs


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


def plot_enable_write_sweep_fine(
    data_list2, save_fig=False, output_path="enable_write_sweep_fine.pdf"
):
    """
    Plots the fine enable write sweep for the provided data list.
    Returns (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_enable_write_sweep_multiple(ax, data_list2)
    ax.set_xlim([260, 310])
    if save_fig:
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


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


def plot_enable_current_vs_temp(
    data, save_fig=False, output_path="enable_current_vs_temp.png"
):
    """
    Plots enable current vs. critical current and channel temperature for all cells.
    Returns (fig, axs, axs2).
    """
    colors = CMAP(np.linspace(0.1, 1, 4))
    markers = ["o", "s", "D", "^"]
    fig, axs = plt.subplots(
        1, 1, figsize=(120 / 25.4, 90 / 25.4), sharex=True, sharey=True
    )
    axs2 = axs.twinx()
    for d in data:
        axs.plot(
            d["xfit"],
            d["yfit"],
            label=f"Cell {d['cell']}",
            color=colors[d["column"]],
            marker=markers[d["row"]],
        )
        axs2.plot(
            d["enable_currents"], d["channel_temperature"], color="grey", marker="o"
        )
    axs2.set_ybound(lower=0)
    axs.legend(loc="upper right")
    axs.set_xlim(0, 600)
    axs.set_ylim(0, 1000)
    axs.set_xlabel("Enable Current ($\mu$A)")
    axs.set_ylabel("Critical Current ($\mu$A)")
    axs2.set_ylabel("Channel Temperature (K)")
    plt.tight_layout()
    if save_fig:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, axs, axs2


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


def plot_read_sweep_write_current(data_list, save_path=None):
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_list, "bit_error_rate", "write_current")
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\mu$A]",
    )
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


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


def plot_combined_dc_figure(axs: List[Axes], dict_list: list) -> List[Axes]:
    """
    Plot combined IV and critical current figures on provided axes.
    """
    axs[0].set_axis_off()
    plot_current_voltage_from_dc_sweep(axs[1], dict_list)
    plot_critical_currents_from_dc_sweep(
        axs[2], dict_list, substrate_temp=PROBE_STATION_TEMP
    )
    axs[1].legend(
        loc="lower right",
        fontsize=5,
        frameon=False,
        handlelength=1,
        handleheight=1,
        borderpad=0.1,
        labelspacing=0.2,
    )
    axs[1].set_box_aspect(1.0)
    axs[2].set_box_aspect(1.0)
    axs[2].set_xlim(-500, 500)
    axs[2].xaxis.set_major_locator(MultipleLocator(250))
    return axs


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

        xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.75)
        plot_linear_fit(ax, xfit, yfit)

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


def plot_ic_vs_ih_array(
    heater_currents,
    avg_current,
    ystd,
    cell_names,
    save_fig=False,
    output_path="ic_vs_ih_array.png",
):
    """
    Plots Ic vs Ih for all cells in the array, including average fit line.
    Returns (fig, ax).
    """
    row_colors = {
        "A": "#1f77b4",
        "B": "#ff7f0e",
        "C": "#2ca02c",
        "D": "#d62728",
        "E": "#9467bd",
        "F": "#8c564b",
        "G": "#e377c2",
    }
    col_linestyles = {
        "1": "-",
        "2": "--",
        "3": "-.",
        "4": ":",
        "5": (0, (3, 1, 1, 1)),
        "6": (0, (5, 2)),
        "7": (0, (1, 1)),
    }
    fig, ax = plt.subplots(figsize=(3, 3))
    x_intercepts = []
    y_intercepts = []
    avg_error_list = []
    rows_in_legend = set()
    # Iterate over each cell
    for j in range(heater_currents.shape[1]):
        ih = np.squeeze(heater_currents[0, j]) * 1e6
        ic = np.squeeze(avg_current[0, j])
        err = np.squeeze(ystd[0, j])

        cell_name = str(cell_names[0, j][0])
        row, col = cell_name[0], cell_name[1]

        color = row_colors.get(row, "black")
        linestyle = col_linestyles.get(col, "-")

        # Add to legend only if the row is not already included
        label = f"{row}" if row not in rows_in_legend else None
        if label:
            rows_in_legend.add(row)

        ax.errorbar(
            ih,
            ic,
            yerr=err,
            label=label,
            color=color,
            linestyle=linestyle,
            marker="o",  # Slightly larger marker
            markersize=2,
            linewidth=1.2,  # Main line width
            elinewidth=1.75,  # Thinner error bar lines
            capsize=2,  # No caps for error bars
            alpha=0.9,  # Slight transparency for overlap
        )
        ax.plot(
            ih,
            ic,
            label=label,
            color=color,
            linestyle=linestyle,
            marker="none",
            linewidth=1.5,  # Main line width
            alpha=0.5,  # Slight transparency for overlap
        )
        avg_error_list.append(np.mean(err))
        # Linear fit for intercepts (200-600 µA)
        valid_indices = (ih >= 200) & (ih <= 550)
        ih_filtered = ih[valid_indices]
        ic_filtered = ic[valid_indices]

        if len(ih_filtered) > 1:
            z = np.polyfit(ih_filtered, ic_filtered, 1)
            x_intercept = -z[1] / z[0]
            y_intercept = z[1]

            x_intercepts.append(x_intercept)
            y_intercepts.append(y_intercept)

    # Average fit line
    filtered_x = np.array(x_intercepts)
    filtered_y = np.array(y_intercepts)
    valid_avg = (filtered_x > 0) & (filtered_x < 1e3)
    avg_x_intercept = np.mean(filtered_x[valid_avg])
    avg_y_intercept = np.mean(filtered_y[valid_avg])

    def avg_line(x):
        slope = avg_y_intercept / avg_x_intercept
        return -slope * x + avg_y_intercept

    fit_range = np.linspace(0, 800, 100)
    ax.plot(
        fit_range,
        avg_line(fit_range),
        color="black",
        linestyle="-",
        linewidth=2,
        label="Fit",
    )

    # Final touches
    ax.set_xlabel(r"$I_{\text{enable}}$ [µA]")
    ax.set_ylabel(r"$I_c$ [µA]")
    # ax.set_title(r"$I_c$ vs. $I_h$ Across Array Cells")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(ncol=2, frameon=False, loc="upper right")
    ax.set_ybound(lower=0)
    ax.set_xlim(0, 800)
    plt.tight_layout()
    if save_fig:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig, ax


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


def plot_wafer_maps(maps, titles, cmaps, grid_x, grid_y, radius, annotate_points=False):
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.5), dpi=300)  # 7.2" ≈ 2-column width
    for ax, title, (grid_z, pts, vals), cmap in zip(axes, titles, maps, cmaps):
        circle = plt.Circle((0, 0), radius, color="k", lw=0.5, fill=False)
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=30, cmap=cmap)
        # ax.scatter(pts[:, 0], pts[:, 1], c='k', s=8, zorder=10)
        if annotate_points:
            for (x, y), v in zip(pts, vals):
                ax.text(
                    x,
                    y,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                    zorder=11,
                )
        ax.add_artist(circle)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        cbar = fig.colorbar(
            contour, ax=ax, orientation="vertical", fraction=0.046, pad=0.04
        )
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Thickness (nm)", fontsize=9)
    plt.tight_layout()
    return fig


def plot_probe_station_prbs(
    data_list,
    trim=4500,
    error_locs=None,
    save_fig=False,
    output_path="probe_station_prbs.pdf",
):
    N = len(data_list)
    cmap = plt.get_cmap("Reds")
    colors = cmap(np.linspace(0.5, 1.0, N))
    fig, ax = plt.subplots(figsize=(6, 2.5))
    for i, data in enumerate(data_list):
        x = data["trace_chan"][0] * 1e6  # µs
        y = data["trace_chan"][1] * 1e3  # mV
        x_trimmed = x[trim:-trim]
        y_trimmed = y[trim:-trim] + (i * 20)
        ax.plot(x_trimmed, y_trimmed, color="black", linewidth=0.75)
        bit_write = "".join(data["bit_string"].flatten())
        bit_read = "".join(data["byte_meas"].flatten())
        errors = [bw != br for bw, br in zip(bit_write, bit_read)]
        text_x = x_trimmed[-1] + 0.5
        text_y = y_trimmed[len(y_trimmed) // 2]
        ax.text(
            text_x, text_y, f"Write: {bit_write}", fontsize=8, va="center", ha="left"
        )
        for j, error in enumerate(errors):
            if error:
                ex = 0.4 + j * 1
                ey = -5 + i * 20
                exw = 0.5
                eyw = 15
                px = [ex, ex + exw, ex + exw, ex]
                py = [ey, ey, ey + eyw, ey + eyw]
                polygon = Polygon(
                    xy=list(zip(px, py)), color=colors[-1], alpha=0.5, linewidth=0
                )
                ax.add_patch(polygon)
    ax.set_xlabel("$t$ [µs]")
    ax.set_ylabel("$V$ [mV]")
    ax.tick_params(direction="in", length=3, width=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_fig:
        fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    return fig, ax


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


def plot_read_current_sweep_enable_read(
    dict_list,
    data_list,
    data_list2,
    save=False,
    output_path="read_current_sweep_enable_read.pdf",
):
    """
    Plot the read and write current/temperature sweeps.
    """
    colors = CMAP(np.linspace(0, 1, len(data_list2)))
    # Preprocess
    read_temperatures = []
    enable_read_currents = []
    for data_dict in dict_list:
        read_temperature = get_channel_temperature(data_dict, "read")
        enable_read_current = get_enable_read_current(data_dict)
        read_temperatures.append(read_temperature)
        enable_read_currents.append(enable_read_current)
    enable_write_currents = []
    write_temperatures = []
    for i, data_dict in enumerate(data_list):
        enable_write_current = get_enable_write_current(data_dict)
        write_temperature = get_channel_temperature(data_dict, "write")
        enable_write_currents.append(enable_write_current)
        write_temperatures.append(write_temperature)
    # Plot
    fig, axs = plt.subplots(
        2, 2, figsize=(6, 3), constrained_layout=True, width_ratios=[1, 0.25]
    )
    ax: plt.Axes = axs[1, 0]
    plot_enable_read_sweep(ax, dict_list[::-1], marker=".")
    ax: plt.Axes = axs[1, 1]
    plot_enable_read_temp(ax, enable_read_currents, read_temperatures)
    ax = axs[0, 0]
    plot_enable_write_sweep(ax, data_list2, marker=".")
    ax = axs[0, 1]
    plot_enable_write_temp(ax, enable_write_currents, write_temperatures)
    if save:
        fig.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_read_current_sweep_three(
    dict_list, save_fig=False, output_path="read_current_sweep_three2.pdf"
):
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.5)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    cax = fig.add_subplot(gs[3])  # dedicated colorbar axis
    for i in range(3):
        plot_read_sweep_array(
            axs[i], dict_list[i], "bit_error_rate", "enable_read_current"
        )
        enable_write_temp = get_channel_temperature(dict_list[i][0], "write")
        plot_fill_between_array(axs[i], dict_list[i])
        axs[i].set_xlim(400, 1000)
        axs[i].set_ylim(0, 1)
        axs[i].set_xlabel("$I_{\\mathrm{read}}$ [µA]")
        axs[i].set_title(
            f"$I_{{EW}}$={290 + i * 10} [µA]\n$T_{{W}}$={enable_write_temp:.2f} [K]",
            fontsize=8,
        )
        axs[i].set_box_aspect(1.0)
        axs[i].xaxis.set_major_locator(plt.MultipleLocator(200))
    axs[0].set_ylabel("BER")
    axpos = axs[2].get_position()
    cbar = add_colorbar(axs[2], dict_list, "enable_read_current", cax=cax)
    cbar.ax.set_position([axpos.x1 + 0.02, axpos.y0, 0.01, axpos.y1 - axpos.y0])
    cbar.set_ticks(plt.MaxNLocator(nbins=6))
    if save_fig:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_read_current_sweep_enable_write(
    data_list,
    data_list2,
    colors,
    save_fig=False,
    output_path="read_current_sweep_enable_write2.pdf",
):
    fig, axs = plt.subplots(
        1, 2, figsize=(8.37, 2), constrained_layout=True, width_ratios=[1, 0.25]
    )
    # Left plot: BER vs. enable_write_current
    ax = axs[0]
    for j, data_dict in enumerate(data_list2):
        plot_read_sweep(
            ax, data_dict, "bit_error_rate", "enable_write_current", color=colors[j]
        )
        plot_fill_between(ax, data_dict, fill_color=colors[j])
        enable_write_current = get_enable_write_current(data_dict)
    ax.set_xlabel("$I_{\\mathrm{read}}$ [$\\mu$A]")
    ax.set_ylabel("BER")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlim(400, 1000)
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    # Right plot: T_write vs. enable_write_current
    ax = axs[1]
    enable_write_currents = []
    write_temperatures = []
    for j, data_dict in enumerate(data_list):
        write_current = get_write_current(data_dict)
        enable_write_current = get_enable_write_current(data_dict)
        write_temperature = get_channel_temperature(data_dict, "write")
        enable_write_currents.append(enable_write_current)
        write_temperatures.append(write_temperature)
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
            markeredgecolor="none",
            markerfacecolor=colors[i],
        )
    ax.set_ylabel("$T_{\\mathrm{write}}$ [K]")
    ax.set_xlabel("$I_{\\mathrm{enable}}$ [$\\mu$A]")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    if save_fig:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_simulation_results(axs, ltsp_data_dict, case=16):
    """Plot simulation results for a given case on provided axes."""
    from nmem.simulation.spice_circuits.plotting import create_plot

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


def plot_read_current_sweep_sim(
    files,
    ltsp_data_dict,
    dict_list,
    write_current_list2,
    save_fig=False,
    output_path="spice_comparison_sim.pdf",
):
    from nmem.simulation.spice_circuits.plotting import create_plot

    inner = [
        ["T0", "T1", "T2", "T3"],
    ]
    innerb = [
        ["B0", "B1", "B2", "B3"],
    ]
    outer_nested_mosaic = [
        [inner],
        [innerb],
    ]
    fig, axs = plt.subplot_mosaic(
        outer_nested_mosaic,
        figsize=(6, 3),
        height_ratios=[1, 0.25],
    )
    CASE = 16
    create_plot(axs, ltsp_data_dict, cases=[CASE])
    case_current = ltsp_data_dict[CASE]["read_current"][CASE]
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
    colors = CMAP(np.linspace(0, 1, len(dict_list)))
    col_set = [colors[i] for i in [0, 2, -1]]
    files_sel = [files[i] for i in [0, 2, -1]]
    max_write_current = 300
    for i, file in enumerate(files_sel):
        data = ltspice.Ltspice(f"data/{file}").parse()
        ltsp_data_dict = process_read_data(data)
    axs["T1"].set_ylabel("")
    axs["T2"].set_ylabel("")
    axs["T3"].set_ylabel("")
    axs["B1"].set_ylabel("")
    axs["B2"].set_ylabel("")
    axs["B3"].set_ylabel("")
    fig.subplots_adjust(hspace=0.6, wspace=0.5)
    fig.patch.set_alpha(0)
    ax_legend = fig.add_axes([0.5, 0.95, 0.1, 0.01])
    ax_legend.axis("off")
    ax_legend.legend(
        selected_handles,
        selected_labels2,
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        handlelength=2.5,
        fontsize=8,
    )
    if save_fig:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_current_sweep_results(files, ltsp_data_dict, dict_list, write_current_list):
    inner = [
        ["T0", "T1", "T2", "T3"],
    ]
    innerb = [
        ["B0", "B1", "B2", "B3"],
    ]
    inner2 = [
        ["A", "B"],
    ]
    inner3 = [
        ["C", "D"],
    ]
    outer_nested_mosaic = [
        [inner],
        [innerb],
        [inner2],
        [inner3],
    ]
    fig, axs = plt.subplot_mosaic(
        outer_nested_mosaic,
        figsize=(180 / 25.4, 180 / 25.4),
        height_ratios=[2, 0.5, 1, 1],
    )

    CASE = 16
    create_plot(axs, ltsp_data_dict, cases=[CASE])
    case_current = ltsp_data_dict[CASE]["read_current"][CASE]

    handles, labels = axs["T0"].get_legend_handles_labels()
    # Select specific items
    selected_labels = [
        "Left Branch Current",
        "Right Branch Current",
        "Left Critical Current",
        "Right Critical Current",
    ]
    selected_labels2 = [
        "$i_{\mathrm{H_L}}$",
        "$i_{\mathrm{H_R}}$",
        "$I_{\mathrm{c,H_L}}$",
        "$I_{\mathrm{c,H_R}}$",
    ]
    selected_handles = [handles[labels.index(lbl)] for lbl in selected_labels]

    plot_read_sweep_array(
        axs["A"],
        dict_list,
        "bit_error_rate",
        "write_current",
        marker=".",
        linestyle="-",
        markersize=4,
    )
    axs["A"].set_xlim(650, 850)
    axs["A"].set_ylabel("BER")
    axs["A"].set_xlabel("$I_{\mathrm{read}}$ [µA]", labelpad=-1)
    plot_read_switch_probability_array(
        axs["B"], dict_list, write_current_list, marker=".", linestyle="-", markersize=2
    )
    axs["B"].set_xlim(650, 850)
    # ax.axvline(IRM, color="black", linestyle="--", linewidth=0.5)
    axs["B"].set_xlabel("$I_{\mathrm{read}}$ [µA]", labelpad=-1)
    axs["D"].set_xlabel("$I_{\mathrm{read}}$ [µA]", labelpad=-1)

    axs["C"].set_xlim(650, 850)
    axs["D"].set_xlim(650, 850)
    axs["C"].set_xlabel("$I_{\mathrm{read}}$ [µA]", labelpad=-1)
    axs["C"].set_ylabel("BER")
    axs["B"].set_ylabel("Switching Probability")
    axs["D"].set_ylabel("Switching Probability")

    colors = CMAP(np.linspace(0, 1, len(dict_list)))
    col_set = [colors[i] for i in [0, 2, -1]]
    files_sel = [files[i] for i in [0, 2, -1]]
    max_write_current = 300
    for i, file in enumerate(files_sel):
        data = ltspice.Ltspice(f"data/{file}").parse()
        ltsp_data_dict = process_read_data(data)
        ltsp_write_current = ltsp_data_dict[0]["write_current"][0]
        plot_current_sweep_ber(
            axs["C"],
            ltsp_data_dict,
            color=CMAP(ltsp_write_current / max_write_current),
            label=f"{ltsp_write_current} $\mu$A",
            marker=".",
            linestyle="-",
            markersize=5,
        )

        plot_current_sweep_switching(
            axs["D"],
            ltsp_data_dict,
            color=CMAP(ltsp_write_current / max_write_current),
            label=f"{ltsp_write_current} $\mu$A",
            marker=".",
            markersize=5,
        )

    axs["A"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)
    axs["B"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)
    axs["C"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)
    axs["D"].axvline(case_current, color="black", linestyle="--", linewidth=0.5)

    axs["B"].legend(
        loc="upper right",
        labelspacing=0.1,
        fontsize=6,
    )
    axs["D"].legend(
        loc="upper right",
        labelspacing=0.1,
        fontsize=6,
    )

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.patch.set_alpha(0)

    ax_legend = fig.add_axes([0.5, 0.9, 0.1, 0.01])
    ax_legend.axis("off")
    ax_legend.legend(
        selected_handles,
        selected_labels2,
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        handlelength=2.5,
        fontsize=8,
    )
    save_fig = False
    if save_fig:
        plt.savefig("spice_comparison.pdf", bbox_inches="tight")
    plt.show()


def plot_read_current_operating(dict_list):
    """Plot all figures using the provided data dictionary list."""
    fig, axs = plt.subplot_mosaic("AB;CD", figsize=(8.3, 4))

    plot_read_sweep_array(
        axs["A"],
        dict_list,
        "bit_error_rate",
        "write_current",
    )
    axs["A"].set_xlim(650, 850)

    plot_read_switch_probability_array(axs["B"], dict_list)
    axs["B"].set_xlim(650, 850)
    axs["A"].set_xlabel("$I_{\\mathrm{read}}$ [$\\mu$A]", labelpad=-3)
    axs["A"].set_ylabel("BER")

    ax = axs["C"]
    for data_dict in dict_list:
        state_current_markers = get_state_current_markers(data_dict, "read_current")
        write_current = get_write_current(data_dict)
        for i, state_current in enumerate(state_current_markers[0, :]):
            if state_current > 0:
                ax.plot(
                    write_current,
                    state_current,
                    "o",
                    label=f"{write_current} $\\mu$A",
                    markerfacecolor=RBCOLORS[i],
                    markeredgecolor="none",
                    markersize=4,
                )
    ax.set_xlim(0, write_current)
    ax.set_ylabel("$I_{\\mathrm{state}}$ [$\\mu$A]")
    ax.set_xlabel("$I_{\\mathrm{write}}$ [$\\mu$A]")

    ic_list = [IRM]
    write_current_list = [0]
    ic_list2 = [IRM]
    write_current_list2 = [0]
    for data_dict in dict_list:
        write_current = get_write_current(data_dict)

        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        read_currents = get_read_currents(data_dict)
        if not np.isnan(berargs[0]) and write_current < 100:
            ic_list.append(read_currents[berargs[0]])
            write_current_list.append(write_current)
        if not np.isnan(berargs[2]) and write_current > 100:
            ic_list.append(read_currents[berargs[3]])
            write_current_list.append(write_current)

        if not np.isnan(berargs[1]):
            ic_list2.append(read_currents[berargs[1]])
            write_current_list2.append(write_current)
        if not np.isnan(berargs[3]):
            ic_list2.append(read_currents[berargs[2]])
            write_current_list2.append(write_current)

    ax.plot(write_current_list, ic_list, "-", color="grey", linewidth=0.5)
    ax.plot(write_current_list2, ic_list2, "-", color="grey", linewidth=0.5)
    ax.set_xlim(0, 300)
    ax.set_ylabel("$I_{\\mathrm{read}}$ [$\\mu$A]")
    ax.set_xlabel("$I_{\\mathrm{write}}$ [$\\mu$A]")
    ax.axhline(IRM, color="black", linestyle="--", linewidth=0.5)
    persistent_current = []
    upper = []
    lower = []
    for i, write_current in enumerate(write_current_list):
        if write_current > IRHL_TR / 2:
            ip = np.abs(write_current - IRHL_TR)
        else:
            ip = write_current
        if ip > IRHL_TR:
            ip = IRHL_TR
    write_current_array = np.linspace(
        write_current_list[0], write_current_list[-1], 1000
    )
    persistent_current = np.where(
        write_current_array > IRHL_TR / 2,
        np.abs(write_current_array - IRHL_TR),
        write_current_array,
    )
    persistent_current = np.where(
        persistent_current > IRHL_TR, IRHL_TR, persistent_current
    )
    upper = IRM + persistent_current / 2
    lower = IRM - persistent_current / 2
    ax.fill_between(write_current_array, lower, upper, color="black", alpha=0.1)
    ic = np.array(ic_list)
    ic2 = np.array(ic_list2)

    read_temperature = calculate_channel_temperature(
        CRITICAL_TEMP,
        SUBSTRATE_TEMP,
        data_dict["enable_read_current"] * 1e6,
        CELLS[data_dict["cell"][0]]["x_intercept"],
    ).flatten()

    delta_read_current = np.subtract(ic2, ic)

    critical_current_channel = calculate_critical_current_temp(
        read_temperature, CRITICAL_TEMP, CRITICAL_CURRENT_ZERO
    )
    critical_current_left = critical_current_channel * WIDTH
    critical_current_right = critical_current_channel * (1 - WIDTH)

    retrap2 = (ic2 - critical_current_right) / critical_current_left

    ax = axs["D"]
    ax.plot(
        write_current_list,
        np.abs(delta_read_current),
        "-o",
        color="black",
        markersize=3.5,
    )
    ax.set_xlabel("$I_{\\mathrm{write}}$ [$\\mu$A]")
    ax.set_ylabel("$|\\Delta I_{\\mathrm{read}}|$ [$\\mu$A]")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 110)
    ax.patch.set_alpha(0)
    ax.set_zorder(1)

    ax2 = ax.twinx()
    ax2.plot(write_current_array, persistent_current, "-", color="grey", zorder=-1)
    ax2.set_ylabel("$I_{\\mathrm{persistent}}$ [$\\mu$A]")
    ax2.set_ylim(0, 110)
    ax2.set_zorder(0)
    ax2.fill_between(
        write_current_array,
        np.zeros_like(write_current_array),
        persistent_current,
        color="black",
        alpha=0.1,
    )
    fig.subplots_adjust(wspace=0.33, hspace=0.4)
    save_fig = False
    if save_fig:
        plt.savefig("read_current_sweep_operating.pdf", bbox_inches="tight")
    plt.show()



def plot_retention(delay_list, bit_error_rate_list):
    """Plot BER vs retention time."""
    fig, axs = plt.subplot_mosaic("A;B", figsize=(3.5, 3.5), constrained_layout=True)
    ax = axs["A"]
    ax.set_aspect("equal")
    sort_index = np.argsort(delay_list)
    delay_list = delay_list[sort_index]
    bit_error_rate_list = bit_error_rate_list[sort_index]
    ax.plot(delay_list, bit_error_rate_list, marker="o", color="black")
    ax.set_ylabel("BER")
    ax.set_xlabel("Memory Retention Time (s)")
    ax.set_xscale("log")
    ax.set_xbound(lower=1e-6)
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")
    ax.grid(True, which="both", linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim([1e-4, 1e-3])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    return fig, axs
