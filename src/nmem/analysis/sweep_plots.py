from typing import Literal
import warnings

import ltspice
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

# Suppress numpy warnings about all-nan slices
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='invalid value encountered')

from nmem.analysis.bit_error import (
    calculate_ber_errorbar,
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_total_switches_norm,
)
from nmem.analysis.constants import (
    CRITICAL_CURRENT_ZERO,
    CRITICAL_TEMP,
    IC0_C3,
    IRHL_TR,
    IRM,
    READ_XMAX,
    READ_XMIN,
    SUBSTRATE_TEMP,
    WIDTH,
)
from nmem.analysis.core_analysis import (
    get_enable_write_width,
    get_fitting_points,
    get_read_width,
    get_write_width,
)
from nmem.analysis.currents import (
    calculate_channel_temperature,
    calculate_critical_current_temp,
    get_channel_temperature,
    get_channel_temperature_sweep,
    get_critical_current_heater_off,
    get_critical_currents_from_trace,
    get_enable_current_sweep,
    get_enable_read_current,
    get_enable_write_current,
    get_read_currents,
    get_state_current_markers,
    get_write_current,
)
from nmem.analysis.plot_utils import (
    add_colorbar,
    plot_fill_between,
    plot_fill_between_array,
    polygon_under_graph,
)
from nmem.analysis.styles import CMAP, CMAP3, RBCOLORS
from nmem.analysis.utils import (
    build_array,
    convert_cell_to_coordinates,
    get_current_cell,
)
from nmem.measurement.cells import (
    CELLS,
)
from nmem.simulation.spice_circuits.functions import process_read_data
from nmem.simulation.spice_circuits.plotting import (
    create_plot,
    plot_current_sweep_ber,
    plot_current_sweep_switching,
)


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

    return ax


def plot_ic_vs_ih_array(
    heater_currents,
    avg_current,
    ystd,
    cell_names,
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
        avg_error_list.append(np.nanmean(err))
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
    
    # Check if we have valid data before calculating means
    if np.any(valid_avg):
        avg_x_intercept = np.nanmean(filtered_x[valid_avg])
        avg_y_intercept = np.nanmean(filtered_y[valid_avg])
    else:
        # Fallback values if no valid data
        avg_x_intercept = 0
        avg_y_intercept = 0

    def avg_line(x):
        if avg_x_intercept != 0:
            slope = avg_y_intercept / avg_x_intercept
            return -slope * x + avg_y_intercept
        else:
            return np.zeros_like(x)  # Return zeros if no valid intercept

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

    return fig, ax


# def plot_enable_write_sweep(ax, dict_list):
#     ax = plot_enable_write_sweep_multiple(ax, dict_list[0:6])
#     ax.set_ylabel("BER")
#     ax.set_xlabel("$I_{\\mathrm{enable}}$ [$\\mu$A]")
#     return ax


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
    dict_list
):
    """
    Plots the enable write sweep for multiple datasets.
    Returns (fig, ax).
    """
    fig, ax = plt.subplots()
    ax = plot_enable_write_sweep_multiple(ax, dict_list)
    ax.set_xlabel("$I_{\\mathrm{enable}}$ [$\\mu$A]")
    ax.set_ylabel("BER")
    return fig, ax


def plot_write_temp_vs_current(
    ax, write_current_array, write_temp_array, critical_current_zero
):
    ax2 = ax.twinx()
    for i in range(4):
        ax.plot(
            write_current_array,
            write_temp_array[:, i],
            linestyle="--",
            marker="o",
            color=RBCOLORS[i],
        )
    limits = ax.get_ylim()
    ic_limits = calculate_critical_current_temp(
        np.array(limits), CRITICAL_TEMP, critical_current_zero
    )
    ax2.set_ylim([ic_limits[0], ic_limits[1]])
    ax2.set_ylabel("$I_{\\mathrm{CH}}$ [$\\mu$A]")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid()
    ax.set_xlabel("$I_{\\mathrm{write}}$ [$\\mu$A]")
    ax.set_ylabel("$T_{\\mathrm{write}}$ [K]")
    return ax, ax2


def plot_enable_write_sweep_fine(
    data_list2
):
    """
    Plots the fine enable write sweep for the provided data list.
    Returns (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_enable_write_sweep_multiple(ax, data_list2)
    ax.set_xlim([260, 310])

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
        ax.fill_between(
            enable_currents,
            bit_error_rate - calculate_ber_errorbar(bit_error_rate),
            bit_error_rate + calculate_ber_errorbar(bit_error_rate),
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
        ax.fill_between(
            read_currents,
            value - calculate_ber_errorbar(value),
            value + calculate_ber_errorbar(value),
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


def plot_read_switch_probability_array(
    ax: Axes, dict_list: list[dict], write_list=None, **kwargs
) -> Axes:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
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


def plot_write_sweep_formatted(ax: plt.Axes, dict_list: list[dict]):
    plot_write_sweep(ax, dict_list)
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(0, 300)
    return ax


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
        data = ltspice.Ltspice(f"../data/ber_sweep_read_current/ltspice_simulation/{file}").parse()
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
        # Ensure berargs has at least 4 elements and handle NaN values properly
        if len(berargs) >= 4:
            if not np.isnan(berargs[0]) and write_current < 100:
                arg_idx = int(berargs[0])
                if 0 <= arg_idx < len(read_currents):
                    ic_list.append(read_currents[arg_idx])
                    write_current_list.append(write_current)
            if not np.isnan(berargs[2]) and write_current > 100:
                arg_idx = int(berargs[3])
                if 0 <= arg_idx < len(read_currents):
                    ic_list.append(read_currents[arg_idx])
                    write_current_list.append(write_current)

            if not np.isnan(berargs[1]):
                arg_idx = int(berargs[1])
                if 0 <= arg_idx < len(read_currents):
                    ic_list2.append(read_currents[arg_idx])
                    write_current_list2.append(write_current)
            if not np.isnan(berargs[3]):
                arg_idx = int(berargs[2])
                if 0 <= arg_idx < len(read_currents):
                    ic_list2.append(read_currents[arg_idx])
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


def plot_retention(delay_list, bit_error_rate_list, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

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
    return fig, ax


def plot_write_sweep_ber(ax, dict_list):
    plot_write_sweep(ax, dict_list)
    ax.set_xlabel("$I_{\\mathrm{write}}$ [µA]")
    ax.set_ylabel("BER")
    ax.set_xlim(0, 300)
    return ax


def plot_temp_vs_current(ax, data, data2):
    ax.plot(
        [d["write_temp"] for d in data],
        [d["write_current"] for d in data],
        "o",
        color="blue",
    )
    ax.plot(
        [d["write_temp"] for d in data2],
        [d["write_current"] for d in data2],
        "o",
        color="red",
    )
    ax.set_xlabel("$T_{\\mathrm{write}}$ [K]")
    ax.set_ylabel("$I_{\\mathrm{ch}}$ [µA]")
    ax.set_ylim(0, 300)
    ax.grid()
    return ax


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


def plot_write_current_enable_sweep_margin(
    dict_list,
    inner,
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

    return fig, axs


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
            if not np.isnan(arg) and arg is not None:
                arg_idx = int(arg)
                if 0 <= arg_idx < len(write_temps) and 0 <= arg_idx < len(enable_currents):
                    write_temp_array[j, i] = write_temps[arg_idx]
                    enable_current_array[j, i] = enable_currents[arg_idx]
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


def plot_write_sweep_formatted_markers(ax: plt.Axes, data_dict: dict):
    data = data_dict.get("data")
    data2 = data_dict.get("data2")
    colors = CMAP(np.linspace(0, 1, 4))
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
    ber_std = calculate_ber_errorbar(bit_error_rate, N)
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


def plot_bit_error_rate_args(ax: Axes, data_dict: dict, color) -> Axes:
    bit_error_rate = get_bit_error_rate(data_dict)
    berargs = get_bit_error_rate_args(bit_error_rate)

    read_current = get_read_currents(data_dict)
    for arg in berargs:
        if not np.isnan(arg) and arg is not None:
            # Convert to int to use as index, with bounds checking
            arg_idx = int(arg)
            if 0 <= arg_idx < len(read_current) and 0 <= arg_idx < len(bit_error_rate):
                ax.plot(
                    read_current[arg_idx],
                    bit_error_rate[arg_idx],
                    marker="o",
                    color=color,
                )
                ax.axvline(read_current[arg_idx], color=color, linestyle="--")
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
