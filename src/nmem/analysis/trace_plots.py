"""
trace_plots.py

Voltage and current trace plotting functions for nmem analysis.
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator, MultipleLocator

from nmem.analysis.constants import PROBE_STATION_TEMP
from nmem.analysis.core_analysis import (
    compute_sigma_separation,
    extract_shifted_traces,
    get_voltage_trace_data,
)
from nmem.analysis.plotting import plot_message
from nmem.analysis.styles import CMAP
from nmem.analysis.sweep_plots import (
    plot_critical_currents_from_dc_sweep,
)


def plot_voltage_trace(
    ax: Axes, time: np.ndarray, voltage: np.ndarray, **kwargs
) -> Axes:
    """Plot a voltage trace on the given axis."""
    ax.plot(time, voltage, **kwargs)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xticklabels([])
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="x", direction="in", which="both")
    return ax


def plot_voltage_trace_zoom(
    ax: Axes, x: np.ndarray, y: np.ndarray, start: float, end: float
) -> Axes:
    xzoom = x[(x > start) & (x < end)]
    yzoom = y[(x > start) & (x < end)]
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
    ax.plot((x - x[0]), y, **kwargs)
    return ax




def plot_voltage_trace_bitstream(ax: Axes, data_dict: dict, trace_name: str) -> Axes:
    x, y = get_voltage_trace_data(data_dict, trace_name)
    ax.plot(x, y, label=trace_name)
    from nmem.analysis.plotting import plot_message

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
    axs[0].set_axis_off()
    plot_current_voltage_from_dc_sweep(axs[1], dict_list)
    from nmem.analysis.plotting import plot_critical_currents_from_dc_sweep

    plot_critical_currents_from_dc_sweep(axs[2], dict_list)
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


def plot_time_concatenated_traces(axs: List[Axes], dict_list: List[dict]) -> List[Axes]:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    colors = np.flipud(colors)
    for idx, data_dict in enumerate(dict_list):
        shift = 10 * idx
        chan_in_x, chan_in_y, enab_in_x, enab_in_y, chan_out_x, chan_out_y = (
            extract_shifted_traces(data_dict, time_shift=shift)
        )
        plot_voltage_trace(axs[0], chan_in_x, chan_in_y, color=colors[0])
        plot_voltage_trace(axs[1], enab_in_x, enab_in_y, color=colors[1])
        plot_voltage_trace(axs[2], chan_out_x, chan_out_y, color=colors[-1])
    axs[2].xaxis.set_major_locator(MultipleLocator(10))
    axs[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    axs[2].set_xlim(0, 50)
    axs[0].legend(["input"], loc="upper right", fontsize=8, frameon=True)
    axs[1].legend(["enable"], loc="upper right", fontsize=8, frameon=True)
    axs[2].legend(["output"], loc="upper right", fontsize=8, frameon=True)
    fig = plt.gcf()
    fig.supylabel("Voltage [mV]", fontsize=9)
    fig.supxlabel("Time [µs]", y=-0.02, fontsize=9)
    fig.subplots_adjust(hspace=0.0)
    return axs


def plot_voltage_pulse_avg(dict_list):
    """Plot voltage pulse traces and histograms."""
    fig, ax_dict = plt.subplot_mosaic("A;B", figsize=(6, 5), constrained_layout=True)
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()

    plot_voltage_trace_averaged(
        ax_dict["A"], dict_list[4], "trace_write_avg", color="#293689", label="Write"
    )
    plot_voltage_trace_averaged(
        ax2, dict_list[4], "trace_ewrite_avg", color="#ff1423", label="Enable\nWrite"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"], dict_list[4], "trace_read0_avg", color="#1966ff", label="Read 0"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"],
        dict_list[4],
        "trace_read1_avg",
        color="#ff7f0e",
        linestyle="--",
        label="Read 1",
    )
    plot_voltage_trace_averaged(
        ax3, dict_list[4], "trace_eread_avg", color="#ff1423", label="Enable\nRead"
    )

    sigma_sep = compute_sigma_separation(dict_list[3], show_print=True)
    ax_dict["A"].legend(loc="upper left", handlelength=1.2)
    ax_dict["A"].set_ylabel("Voltage [mV]")
    ax2.legend(loc="upper right", handlelength=1.2)
    ax2.set_ylabel("Voltage [mV]")
    ax3.legend(loc="upper right", handlelength=1.2)
    ax3.set_ylabel("Voltage [mV]")
    ax_dict["B"].set_xlabel("time [µs]")
    ax_dict["B"].set_ylabel("Voltage [mV]")
    ax_dict["B"].legend(loc="upper left", handlelength=1.2)
    save_fig = False
    if save_fig:
        plt.savefig("voltage_trace_out.png", bbox_inches="tight")
    plt.show()





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




def plot_voltage_pulse_avg(dict_list):
    """Plot voltage pulse traces and histograms."""
    fig, ax_dict = plt.subplot_mosaic("A;B", figsize=(6, 5), constrained_layout=True)
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()

    plot_voltage_trace_averaged(
        ax_dict["A"], dict_list[4], "trace_write_avg", color="#293689", label="Write"
    )
    plot_voltage_trace_averaged(
        ax2, dict_list[4], "trace_ewrite_avg", color="#ff1423", label="Enable\nWrite"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"], dict_list[4], "trace_read0_avg", color="#1966ff", label="Read 0"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"],
        dict_list[4],
        "trace_read1_avg",
        color="#ff7f0e",
        linestyle="--",
        label="Read 1",
    )
    plot_voltage_trace_averaged(
        ax3, dict_list[4], "trace_eread_avg", color="#ff1423", label="Enable\nRead"
    )

    sigma_sep = compute_sigma_separation(dict_list[3], show_print=True)
    ax_dict["A"].legend(loc="upper left", handlelength=1.2)
    ax_dict["A"].set_ylabel("Voltage [mV]")
    ax2.legend(loc="upper right", handlelength=1.2)
    ax2.set_ylabel("Voltage [mV]")
    ax3.legend(loc="upper right", handlelength=1.2)
    ax3.set_ylabel("Voltage [mV]")
    ax_dict["B"].set_xlabel("time [µs]")
    ax_dict["B"].set_ylabel("Voltage [mV]")
    ax_dict["B"].legend(loc="upper left", handlelength=1.2)
    save_fig = False
    if save_fig:
        plt.savefig("voltage_trace_out.png", bbox_inches="tight")
    plt.show()




def plot_time_concatenated_traces(axs: List[Axes], dict_list: List[dict]) -> List[Axes]:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    colors = np.flipud(colors)

    for idx, data_dict in enumerate(dict_list):
        shift = 10 * idx  # Shift time window by 10 µs per dataset
        chan_in_x, chan_in_y, enab_in_x, enab_in_y, chan_out_x, chan_out_y = (
            extract_shifted_traces(data_dict, time_shift=shift)
        )

        plot_voltage_trace(axs[0], chan_in_x, chan_in_y, color=colors[0])
        plot_voltage_trace(axs[1], enab_in_x, enab_in_y, color=colors[1])
        plot_voltage_trace(axs[2], chan_out_x, chan_out_y, color=colors[-1])

    axs[2].xaxis.set_major_locator(MultipleLocator(10))
    axs[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    axs[2].set_xlim(0, 50)
    axs[0].legend(
        ["input"],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    axs[1].legend(
        ["enable"],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    axs[2].legend(
        ["output"],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    fig = plt.gcf()
    fig.supylabel("Voltage [mV]", fontsize=9)
    fig.supxlabel("Time [µs]", y=-0.02, fontsize=9)
    fig.subplots_adjust(hspace=0.0)

    return axs



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
