import ltspice
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Literal
import scipy.io as sio
from nmem.analysis.analysis import import_directory, filter_first
from nmem.simulation.spice_circuits.spice_data_processing import (
    process_read_data,
    get_sweep_parameter,
)

CMAP = plt.get_cmap("coolwarm")

FILL_WIDTH = 5
VOUT_YMAX = 40
VOLTAGE_THRESHOLD = 2.0e-3


def plot_tran_data(
    ax: plt.Axes,
    ltspice_data: ltspice.Ltspice,
    signal_name: str,
    case: int = 0,
    scale: float = 1e6,
    **kwargs,
) -> plt.Axes:
    time = ltspice_data.get_time(case=case)
    signal = ltspice_data.get_data(signal_name, case=case)
    signal_out = signal * scale if signal is not None else np.zeros_like(time)
    ax.plot(time, signal_out, label=signal_name, **kwargs)
    return ax


def plot_current_transient(
    ax: plt.Axes, data_dict: dict, cases=[0], side: Literal["left", "right"] = "left"
) -> plt.Axes:
    for i in cases:
        data = data_dict[i]
        time = data["time"]
        if side == "left":
            left_critical_current = data["tran_left_critical_current"]
            left_branch_current = data["tran_left_branch_current"]
            ax.plot(
                time, left_critical_current, label="Left Critical Current", color="grey"
            )
            ax.plot(
                time, left_branch_current, label="Left Branch Current", color="blue"
            )
        if side == "right":
            right_critical_current = data["tran_right_critical_current"]
            right_branch_current = data["tran_right_branch_current"]
            ax.plot(
                time,
                right_critical_current,
                label="Right Critical Current",
                color="grey",
            )
            ax.plot(
                time, right_branch_current, label="Right Branch Current", color="blue"
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (uA)")
    ax.legend()
    return ax


def plot_branch_fill(
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
) -> plt.Axes:
    if len(data_dict) > 1:
        data_dict = data_dict[0]
    sweep_param = get_sweep_parameter(data_dict)
    sweep_current = data_dict[sweep_param]
    read_zero_voltage = data_dict["read_zero_voltage"]
    read_one_voltage = data_dict["read_one_voltage"]
    ax.plot(sweep_current, read_zero_voltage * 1e3, "-o", label="Read 0")
    ax.plot(sweep_current, read_one_voltage * 1e3, "-o", label="Read 1")
    ax.legend()
    ax.set_ylabel("Output Voltage (mV)")
    ax.set_xlabel(f"{sweep_param} (uA)")
    return ax


def plot_enable_write_current_output(
    ax: plt.Axes,
    data_dict: dict,
) -> plt.Axes:
    enable_write_current = data_dict["enable_write_current"]
    read_output = data_dict["read_output"]
    ax.plot(enable_write_current, read_output[:, 0] * 1e3, "-o", label="Read 0")
    ax.plot(enable_write_current, read_output[:, 1] * 1e3, "-o", label="Read 1")
    ax.legend()
    ax.set_ylabel("Output Voltage (mV)")
    ax.set_xlabel("Enable Write Current (uA)")
    return ax


if __name__ == "__main__":

    l = ltspice.Ltspice("nmem_cell_read.raw")
    l.parse()
    data_dict = process_read_data(l)
    fig, ax = plt.subplots()
    plot_current_sweep_output(ax, data_dict)
    plt.show()

    CASE = 2
    fig, ax = plt.subplots()
    plot_current_transient(ax, data_dict, cases=[CASE])
    plot_branch_fill(ax, data_dict, cases=[CASE])
    fig, ax = plt.subplots()
    plot_current_transient(ax, data_dict, cases=[CASE], side="right")
    plot_branch_fill(ax, data_dict, cases=[CASE], side="right")
