from typing import Literal

import ltspice
import matplotlib.pyplot as plt
import numpy as np

from nmem.simulation.spice_circuits.functions import (
    get_step_parameter,
    process_read_data,
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
    **kwargs,
) -> plt.Axes:
    if len(data_dict) > 1:
        data_dict = data_dict[0]
    sweep_param = get_step_parameter(data_dict)
    sweep_current = data_dict[sweep_param]
    read_zero_voltage = data_dict["read_zero_voltage"]
    read_one_voltage = data_dict["read_one_voltage"]

    base_label = f" {kwargs['label']}" if 'label' in kwargs else ""
    kwargs.pop("label", None)
    ax.plot(sweep_current, read_zero_voltage * 1e3, "-o", label=f"{base_label} Read 0", **kwargs)
    ax.plot(sweep_current, read_one_voltage * 1e3, "--o", label=f"{base_label} Read 1", **kwargs)
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

    base_label = f" {kwargs['label']}" if 'label' in kwargs else ""
    kwargs.pop("label", None)
    ax.plot(sweep_current, ber , "-o", label=f"{base_label}", **kwargs)
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

    base_label = f" {kwargs['label']}" if 'label' in kwargs else ""
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


if __name__ == "__main__":

    l = ltspice.Ltspice("nmem_cell_read.raw")
    l.parse()
    data_dict = process_read_data(l)
    fig, ax = plt.subplots()
    plot_current_sweep_output(ax, data_dict)
    plt.show()

    CASE = 4
    fig, ax = plt.subplots()
    plot_current_transient(ax, data_dict, cases=[CASE])
    plot_branch_fill(ax, data_dict, cases=[CASE])
    fig, ax = plt.subplots()
    plot_current_transient(ax, data_dict, cases=[CASE], side="right")
    plot_branch_fill(ax, data_dict, cases=[CASE], side="right")

