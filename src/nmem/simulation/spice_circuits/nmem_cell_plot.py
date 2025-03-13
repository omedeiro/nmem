import ltspice
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
import scipy.io as sio
from nmem.analysis.analysis import import_directory, filter_first
from nmem.simulation.spice_circuits.spice_data_processing import (
    get_persistent_current,
    get_write_current,
    get_max_output,
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
    signal = ltspice_data.get_data(signal_name, case=case) * scale
    ax.plot(time, signal, label=signal_name, **kwargs)
    return ax


def plot_nmem_cell(ax: plt.Axes, file_path: str) -> plt.Axes:
    l = ltspice.Ltspice(file_path).parse()
    colors = [CMAP(i) for i in np.linspace(0, 1, l.case_count)]
    for i in range(l.case_count):
        ax = plot_tran_data(ax, l, "Ix(HL:drain)", case=i, color=colors[i])
        ax = plot_tran_data(
            ax, l, "Ix(HR:drain)", case=i, color=colors[i], linestyle="--"
        )
        ax = plot_tran_data(ax, l, "V(ichl)", case=i, color="k", linestyle="-.")
        ax = plot_tran_data(ax, l, "V(ichr)", case=i, color="k", linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    return ax


def plot_read_current_output(
    ax: plt.Axes,
    data_dict: dict,
) -> plt.Axes:
    read_current = data_dict["read_current"]
    read_output = data_dict["read_output"]
    ax.plot(read_current, read_output[:, 0] * 1e3, "-o", label="Read 0")
    ax.plot(read_current, read_output[:, 1] * 1e3, "-o", label="Read 1")
    ax.legend()
    ax.set_ylabel("Output Voltage (mV)")
    ax.set_xlabel("Read Current (uA)")
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
    read_outputs = data_dict["read_output"]
    read_current = data_dict["read_current"]
    read_current = read_current.reshape(-1, 1)
    write_current = data_dict["write_current"]
    write_one_voltage = data_dict["write_one_voltage"]
    read_output = data_dict["read_output"]
    persistent_current = data_dict["persistent_current"]
    read_zero_voltage = data_dict["read_zero"]
    read_one_voltage = data_dict["read_one"]
    enable_write_current = data_dict["enable_write_current"]
    fig, ax = plt.subplots()
    sweep_param = write_current
    plot_read_current_output(ax, data_dict)

    nominal_region = (read_zero_voltage < VOLTAGE_THRESHOLD) & (
        read_one_voltage > VOLTAGE_THRESHOLD
    )
    inverting_region = (read_zero_voltage > VOLTAGE_THRESHOLD) & (
        read_one_voltage < VOLTAGE_THRESHOLD
    )
    ber = np.ones_like(sweep_param) * 0.5
    ber[nominal_region] = 0
    ber[inverting_region] = 1
    ylim = ax.get_ylim()
    sweep_param_nominal = sweep_param[nominal_region]
    sweep_param_inverting = sweep_param[inverting_region]
    if len(sweep_param_nominal) > 0:
        for wc in sweep_param_nominal:
            ax.fill_betweenx(
                ylim,
                wc - FILL_WIDTH,
                wc + FILL_WIDTH,
                color="g",
                alpha=0.3,
                label="_Nominal Region",
            )
    if len(sweep_param_inverting) > 0:
        for wc in sweep_param_inverting:
            ax.fill_betweenx(
                ylim,
                wc - FILL_WIDTH,
                wc + FILL_WIDTH,
                color="r",
                alpha=0.3,
                label="_Inverting Region",
            )
    ax.legend()
    ax.plot(sweep_param, write_one_voltage * 1e3, "-o", label="Write One Voltage")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(sweep_param, persistent_current, "-o", label="Persistent Current")
    ax.set_xlabel("Write Current (uA)")
    ax.set_ylabel("Persistent Current (uA)")
    X_SHIFT = 236  # ICHL
    ax.plot([0, ax.get_ylim()[1] * 2], [0, ax.get_ylim()[1]], "--", label="y=x/2")
    ax.plot(
        [X_SHIFT, ax.get_ylim()[1] * 2 + X_SHIFT],
        [0, ax.get_ylim()[1]],
        "--",
        label="_y=x/2",
    )
    ax2 = ax.twinx()
    ax2.plot(sweep_param, ber, "-o", label="Bit Error Rate", color="r")
    ax.legend()
