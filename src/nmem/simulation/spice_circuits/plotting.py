from typing import Literal

import ltspice
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio.v2 as imageio
import matplotlib.font_manager as fm
from nmem.simulation.spice_circuits.functions import (
    get_step_parameter,
    process_read_data,
)
import matplotlib.pyplot as plt
from cycler import cycler

CMAP = plt.get_cmap("coolwarm")
if os.name == "Windows":
    font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"
if os.name == "posix":
    font_path = "/home/omedeiro/Inter-VariableFont_opsz,wght.ttf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

FILL_WIDTH = 5
VOUT_YMAX = 40
VOLTAGE_THRESHOLD = 2.0e-3

plt.rcParams.update(
    {
        "figure.figsize": [3.5, 3.5],
        "font.size": 7,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.family": "Inter",
        "lines.markersize": 2,
        "lines.linewidth": 1.2,
        "legend.fontsize": 5,
        "legend.frameon": False,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)


COLORS = [
    "#1b9e77",  # Teal green
    "#d95f02",  # Burnt orange
    "#7570b3",  # Muted blue-purple
    "#e7298a",  # Reddish pink
    "#66a61e",  # Olive green
    "#e6ab02",  # Mustard yellow
    "#a6761d",  # Brown
    "#666666",  # Dark gray
]

plt.rcParams["axes.prop_cycle"] = cycler(color=COLORS)


def plot_transient(
    ax: plt.Axes,
    data_dict: dict,
    cases=[0],
    signal_name: str = "tran_left_critical_current",
    **kwargs,
) -> plt.Axes:
    for i in cases:
        data = data_dict[i]
        time = data["time"]
        signal = data[signal_name]
        ax.plot(time, signal, label=f"{signal_name}", **kwargs)
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel(f"{signal_name}")
    # ax.legend(loc="upper right")
    return ax


def plot_transient_fill(
    ax: plt.Axes,
    data_dict: dict,
    cases=[0],
    s1: str = "tran_left_critical_current",
    s2: str = "tran_left_branch_current",
    **kwargs,
) -> plt.Axes:
    for i in cases:
        data = data_dict[i]
        time = data["time"]
        signal1 = data[s1]
        signal2 = data[s2]
        ax.fill_between(
            time,
            signal2,
            signal1,
            color=CMAP(0.5),
            alpha=0.5,
            label="Left Branch",
            **kwargs,
        )
    return ax


def plot_transient_fill_branch(
    ax: plt.Axes, data_dict: dict, cases=[0], side: Literal["left", "right"] = "left"
) -> plt.Axes:
    for i in cases:
        data_dict = data_dict[i]
        time = data_dict["time"]
        if side == "left":
            left_critical_current = data_dict["tran_left_critical_current"]
            left_branch_current = data_dict["tran_left_branch_current"]
            ax.fill_between(
                time,
                left_branch_current,
                left_critical_current,
                color=CMAP(0.5),
                alpha=0.5,
                label="Left Branch",
            )
        if side == "right":
            right_critical_current = data_dict["tran_right_critical_current"]
            right_branch_current = data_dict["tran_right_branch_current"]
            ax.fill_between(
                time,
                right_branch_current,
                right_critical_current,
                color=CMAP(0.5),
                alpha=0.5,
                label="Right Branch",
            )
    return ax


def plot_current_sweep_output(
    ax: plt.Axes,
    data_dict: dict,
    **kwargs,
) -> plt.Axes:
    if len(data_dict) > 1:
        data_dict = data_dict[0]
    sweep_param = get_step_parameter(data_dict)
    sweep_current = data_dict[sweep_param]
    read_zero_voltage = data_dict["read_zero_voltage"]
    read_one_voltage = data_dict["read_one_voltage"]

    base_label = f" {kwargs['label']}" if "label" in kwargs else ""
    kwargs.pop("label", None)
    ax.plot(
        sweep_current,
        read_zero_voltage * 1e3,
        "-o",
        label=f"{base_label} Read 0",
        **kwargs,
    )
    ax.plot(
        sweep_current,
        read_one_voltage * 1e3,
        "--o",
        label=f"{base_label} Read 1",
        **kwargs,
    )
    ax.set_ylabel("Output Voltage (mV)")
    ax.set_xlabel(f"{sweep_param} (uA)")
    return ax


def plot_current_sweep_ber(
    ax: plt.Axes,
    data_dict: dict,
    **kwargs,
) -> plt.Axes:
    if len(data_dict) > 1:
        data_dict = data_dict[0]
    sweep_param = get_step_parameter(data_dict)
    sweep_current = data_dict[sweep_param]
    ber = data_dict["bit_error_rate"]

    base_label = f" {kwargs['label']}" if "label" in kwargs else ""
    kwargs.pop("label", None)
    ax.plot(sweep_current, ber, "-o", label=f"{base_label}", **kwargs)
    ax.set_ylabel("BER")
    ax.set_xlabel(f"{sweep_param} (uA)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    return ax


def plot_current_sweep_persistent(
    ax: plt.Axes,
    data_dict: dict,
    **kwargs,
) -> plt.Axes:
    if len(data_dict) > 1:
        data_dict = data_dict[0]
    sweep_param = get_step_parameter(data_dict)
    sweep_current = data_dict[sweep_param]
    persistent_current = data_dict["persistent_current"]

    base_label = f" {kwargs['label']}" if "label" in kwargs else ""
    kwargs.pop("label", None)
    ax.plot(sweep_current, persistent_current, "-o", label=f"{base_label}", **kwargs)
    ax.set_ylabel("Persistent Current (uA)")
    ax.set_xlabel(f"{sweep_param} (uA)")
    return ax


def plot_retrapping_ratio(ax: plt.axes, data_dict: dict, cases: list = [0]) -> plt.Axes:
    for i in cases:
        data = data_dict[i]
        time = data["time"]
        left_branch_critical_current = data["tran_left_critical_current"]
        right_branch_critical_current = data["tran_right_critical_current"]
        left_retrapping_current = data["tran_left_retrapping_current"]
        right_retrapping_current = data["tran_right_retrapping_current"]
        retrapping_ratio = left_retrapping_current / left_branch_critical_current
        ax.plot(time, retrapping_ratio, label="Retrapping Ratio")


def plot_case(ax, data_dict, case, signal_name="left", color=None):
    if color is None:
        if signal_name == "left":
            color = "C0"
        elif signal_name == "right":
            color = "C1"

    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name=f"tran_{signal_name}_critical_current",
        linestyle="--",
        color=color,
    )
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name=f"tran_left_branch_current",
        color="C0",
    )
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name=f"tran_right_branch_current",
        color="C1",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)


def plot_case2(ax, data_dict, case, signal_name="left", color=None):
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name=f"tran_left_branch_current",
        color=color,
    )
    plot_transient(
        ax,
        data_dict,
        cases=[case],
        signal_name=f"tran_right_branch_current",
        color=color,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)


def plot_case_vout(ax, data_dict, case, signal_name, **kwargs):
    plot_transient(ax, data_dict, cases=[case], signal_name=f"{signal_name}", **kwargs)


def plot_panes() -> None:
    ltsp = ltspice.Ltspice(
        "spice_simulation_raw/step_read_current/nmem_cell_read.raw",
    ).parse()
    # colors = plt.cm.viridis(np.linspace(0, 1, 10))
    data_dict = process_read_data(ltsp)
    print(ltsp.case_count)
    colors = plt.cm.viridis(np.linspace(0, 1, ltsp.case_count))
    fig, axs = plt.subplots(2, 4, figsize=(6, 3))
    time_windows = {
        0: (100e-9, 150e-9),
        1: (200e-9, 250e-9),
        2: (300e-9, 350e-9),
        3: (400e-9, 450e-9),
    }
    for case in range(ltsp.case_count):
        for i, time_window in time_windows.items():
            init_case = 0
            sweep_param = data_dict[case]["read_current"]
            sweep_param = sweep_param[case]
            ax: plt.Axes = axs[0, i]
            plot_case(ax, data_dict, init_case, "left")
            plot_case(ax, data_dict, init_case, "right")

            ax.set_ylim(-150, 900)
            ax.set_xlim(time_window)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
            ax.xaxis.set_major_locator(plt.MultipleLocator(10e-9))
            ax.yaxis.set_major_locator(plt.MultipleLocator(500))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(100))
            ax.set_xticklabels([])
            if i != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("I ($\mu$A)")

            # ax.text(0.1, 0.8, f"read current {sweep_param}uA", transform=ax.transAxes)
            ax: plt.Axes = axs[1, i]
            plot_case_vout(
                ax, data_dict, case, "tran_output_voltage", color=COLORS[case]
            )
            ax.set_ylim(-1e-3, 40e-3)
            ax.set_xlim(time_window)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
            ax.yaxis.set_major_locator(plt.MultipleLocator(10e-3))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e3:.0f}"))
            if i != 0:
                ax.set_yticklabels([])
            else:
                ax.set_xlabel("Time (ns)")
                ax.set_ylabel("V (mV)")
            ax.xaxis.set_major_locator(plt.MultipleLocator(50e-9))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(10e-9))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e9:.0f}"))

    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()


if __name__ == "__main__":
    plot_panes()