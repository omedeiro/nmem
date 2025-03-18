from typing import Literal

import ltspice
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio

from nmem.simulation.spice_circuits.functions import (
    get_step_parameter,
    process_read_data,
)

CMAP = plt.get_cmap("coolwarm")

FILL_WIDTH = 5
VOUT_YMAX = 40
VOLTAGE_THRESHOLD = 2.0e-3


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
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{signal_name}")
    ax.legend()
    return ax


def plot_transient_fill(
    ax: plt.Axes,
    data_dict: dict,
    cases=[0],
    s1: str = "tran_left_critical_current",
    s2: str = "tran_left_branch_current",
) -> plt.Axes:
    for i in cases:
        data = data_dict[i]
        time = data["time"]
        signal1 = data[s1]
        signal2 = data[s2]
        ax.fill_between(
            time, signal2, signal1, color=CMAP(0.5), alpha=0.5, label="Left Branch"
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



if __name__ == "__main__":
    ltsp = ltspice.Ltspice("spice_simulation_raw/write_current_sweep/nmem_cell_write_200uA.raw").parse()
    data_dict = process_read_data(ltsp)

    frame_path = os.path.join(os.getcwd(), "spice_simulation_raw", "write_current_sweep", "frames")
    os.makedirs(frame_path, exist_ok=True)
    frame_filenames = []

    for case in range(0, ltsp.case_count, 20):
        write_current = data_dict[case]["write_current"]
        write_current = write_current[case]
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        ax: plt.Axes = axs[0]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_left_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_left_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_left_critical_current",
            s2="tran_left_branch_current",
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax: plt.Axes = axs[1]
        plot_transient(
            ax, data_dict, cases=[case], signal_name="tran_right_critical_current"
        )
        plot_transient(
            ax,
            data_dict,
            cases=[case],
            signal_name="tran_right_branch_current",
            color="grey",
        )
        plot_transient_fill(
            ax,
            data_dict,
            cases=[case],
            s1="tran_right_critical_current",
            s2="tran_right_branch_current",
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.text(0.1, 0.8, f"write current {write_current}uA", transform=ax.transAxes)
        frame_filename = f"{frame_path}/frame_{case}.png"
        plt.savefig(frame_filename)
        frame_filenames.append(frame_filename)
        plt.close(fig)

    # Create GIF
    save_gif = False
    if save_gif:
        gif_filename = frame_path + "/write_current_sweep.gif"
        with imageio.get_writer(gif_filename, mode="I", duration=0.2, loop=0) as writer:
            for filename in frame_filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        print(f"GIF saved as {gif_filename}")