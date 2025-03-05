from typing import Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    CRITICAL_TEMP,
    RETRAP,
    WIDTH,
    ALPHA,
    calculate_branch_currents,
    calculate_critical_current_temp,
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_channel_temperature,
    get_channel_temperature_sweep,
    get_critical_current_heater_off,
    get_critical_current_intercept,
    get_enable_current_sweep,
    get_read_current,
    get_write_current,
    import_directory,
    plot_enable_write_sweep_multiple,
    plot_write_sweep,
)

from nmem.analysis.write_current_sweep_sub import (
    calculate_persistent_currents,
)

IWRITE_XLIM = 100
IWRITE_XLIM_2 = 300
MAX_PERSISTENT = 50
IREAD = 727.5

RBCOLORS = {0: "black", 1: "black", 2: "grey", 3: "grey"}


def get_write_array(write_current_list, num_points=100):
    return np.linspace(0, write_current_list[-1], num_points)


def plot_write_current_sweep_markers(
    ax: plt.Axes, write_temp: np.ndarray, write_current: np.ndarray, color="black"
):
    ax.plot(
        write_temp,
        write_current,
        "--o",
        color=color,
    )

    ax.set_xlabel("$T_{\mathrm{write}}$ [K]")
    ax.set_ylabel("$I_{\mathrm{ch}}$ [$\mu$A]")
    # ax.set_ylim(0, IWRITE_XLIM_2)

    return ax


def process_dict_list(dict_list):
    write_temp_array = np.empty((len(dict_list), 4))
    write_current_array = np.empty((len(dict_list), 1))
    write_channel_array = np.empty((len(dict_list), 4))

    for j, data_dict in enumerate(dict_list):
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_current = get_write_current(data_dict)

        write_temps = get_channel_temperature_sweep(data_dict)
        write_critical_currents = calculate_critical_current_temp(
            write_temps,
            CRITICAL_TEMP,
            get_critical_current_heater_off(data_dict),
        )
        write_current_array[j] = write_current
        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                write_temp_array[j, i] = write_temps[arg]
                write_channel_array[j, i] = calculate_critical_current_temp(
                    write_temps[arg],
                    CRITICAL_TEMP,
                    get_critical_current_heater_off(data_dict),
                )
    return (
        write_current_array,
        write_temp_array,
        write_channel_array,
        write_critical_currents,
    )


def get_branch_currents(data_dict, operation: Literal["read", "write"]):
    read_temperature = np.array([get_channel_temperature(data_dict, operation)])
    critical_current_zero = get_critical_current_intercept(data_dict)
    channel_current_zero = calculate_critical_current_temp(
        read_temperature,
        CRITICAL_TEMP,
        critical_current_zero,
    )
    ichl, irhl, ichr, irhr = calculate_branch_currents(
        read_temperature, CRITICAL_TEMP, RETRAP, WIDTH, channel_current_zero
    )
    return ichl, irhl, ichr, irhr


def plot_write_array(
    ax: plt.Axes, write_current_array, write_channel_array, color="black"
):
    ax.plot(
        write_current_array,
        write_channel_array,
        linestyle="-",
        marker="o",
        color=color,
    )
    ax.set_xlim(0, IWRITE_XLIM)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))

    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$I_{\mathrm{branch}}$ [$\mu$A]")
    ax.grid(axis="x")
    ax.set_aspect(1 / ax.get_data_ratio())
    return ax


def get_write_current_array(dict_list: list[dict]) -> Tuple[np.ndarray, np.ndarray]:
    write_temp_array = np.empty((len(dict_list), 4))
    write_channel_array = np.empty((len(dict_list), 4))
    for j, data_dict in enumerate(dict_list):
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)

        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                write_temp_array[j, i] = get_channel_temperature(data_dict, "write")
                write_channel_array[j, i] = get_enable_current_sweep(data_dict)[arg]
    write_channel_array = np.where(write_temp_array < 1, np.nan, write_channel_array)

    return write_channel_array, write_temp_array


def plot_measured_state_currents(
    ax: plt.Axes,
    write_current_array,
    write_channel_array,
    left_branch_current,
    right_branch_current,
    persistent_currents,
):
    for i in range(4):
        plot_write_array(
            ax, write_current_array, write_channel_array[:, i], color=RBCOLORS[i]
        )

    ax.axhline(left_branch_current, color="green", linestyle="--", label="i_L")
    ax.axhline(
        right_branch_current,
        color="blue",
        linestyle="--",
        label="i_R",
    )
    write_currents = get_write_array(write_current_array)
    ax.plot(
        write_currents,
        right_branch_current + persistent_currents,
        color="blue",
        linestyle="-",
    )
    ax.plot(
        write_currents,
        right_branch_current - persistent_currents,
        color="blue",
        linestyle="-",
    )

    ax.plot(
        write_currents,
        left_branch_current + persistent_currents,
        color="green",
        linestyle="-",
    )
    ax.plot(
        write_currents,
        left_branch_current - persistent_currents,
        color="green",
        linestyle="-",
    )
    return ax


def calculate_branch_currents_read(read_currents, alpha):
    left_branch_current = read_currents * alpha
    right_branch_current = read_currents * (1 - alpha)
    return left_branch_current, right_branch_current


def plot_write_current_sweep(ax, write_sweep_dict_list):
    ax = plot_write_sweep(ax, write_sweep_dict_list)
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(0, IWRITE_XLIM_2)

    ax.axvline(irhl, color="C0", linestyle="--", label="_irhl")
    ax.axvline(irhr, color="C1", linestyle="--", label="_irhr")
    ax.axvline(ichl, color="C2", linestyle="--", label="_ichl")
    ax.axvline(ichr, color="C3", linestyle="--", label="_ichr")

    ax.plot(write_currents, persistent_currents / irhl, color="black", linestyle="--")
    return ax


def plot_write_markers_temp(ax, write_sweep_temp, write_sweep_switch):
    ax = plot_write_current_sweep_markers(
        ax, write_sweep_temp[:, 0], write_sweep_switch[:, 0], color="black"
    )
    ax = plot_write_current_sweep_markers(
        ax, write_sweep_temp[:, 2], write_sweep_switch[:, 2], color="grey"
    )

    ax.axhline(10, color="C2", linestyle="--", label="_ichl")
    ax.axhline(110, color="C2", linestyle="--", label="_ichl")

    return ax


def plot_persistent_current_est(ax: plt.Axes, persistent_current, irhl):
    ax.plot(
        get_write_array(write_current_array),
        persistent_current,
        color="black",
        linestyle="-",
        label="I_P",
    )
    ax.axvline(irhl, color="black", linestyle="--", label="irhl")

    ax.set_xlim(0, IWRITE_XLIM)
    ax.set_ylim(0, MAX_PERSISTENT)
    ax.set_aspect(1 / ax.get_data_ratio())
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$I_{\mathrm{P}}$ [$\mu$A]")
    ax.legend()
    return ax


def plot_sub_figures():
    fig, axs = plt.subplots(1, 2, figsize=(6, 4), constrained_layout=True)
    plot_measured_state_currents(
        axs[0],
        write_current_array,
        write_channel_array,
        left_branch_current,
        right_branch_current,
        persistent_currents,
    )

    plot_persistent_current_est(axs[1], persistent_currents, irhl)

    save = True
    if save:
        fig.savefig("write_current_sweep_sub.pdf", bbox_inches="tight")
    plt.show()


def plot_write_operation_figure():
    fig, axs = plt.subplot_mosaic(
        "AB;CD", figsize=(8.3, 4), width_ratios=[1, 0.25], constrained_layout=True
    )

    ax = axs["A"]
    ax, ax2 = plot_enable_write_sweep_multiple(ax, enable_sweep_dict_list[0:6])
    ax.set_ylabel("BER")
    ax2.set_xlabel("$T_{\mathrm{write}}$ [K]")
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    ax.grid()

    [
        plot_write_array(
            axs["B"], write_current_array, write_channel_array[:, i], color=RBCOLORS[i]
        )
        for i in range(4)
    ]

    # Write current sweep
    plot_write_current_sweep(axs["C"], write_sweep_dict_list)

    plot_write_markers_temp(axs["D"], write_sweep_temp, write_sweep_switch)

    fig.savefig("write_current_sweep_operation.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # Import
    enable_sweep_dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )

    write_sweep_dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\write_current_sweep_enable_write\data"
    )
    write_sweep_dict_list = write_sweep_dict_list[1:]

    # Preprocess
    (
        write_current_array,
        write_temp_array,
        write_channel_array,
        critical_current_zero,
    ) = process_dict_list(enable_sweep_dict_list)

    read_current = get_read_current(enable_sweep_dict_list[0])
    ichl, irhl, ichr, irhr = get_branch_currents(enable_sweep_dict_list[0], "read")
    ichl_write, _, _, _ = get_branch_currents(enable_sweep_dict_list[0], "read")

    write_sweep_markers, write_temperature_markers = get_write_current_array(
        enable_sweep_dict_list
    )

    write_currents = get_write_array(write_current_array)
    persistent_currents = calculate_persistent_currents(write_currents, irhl)
    write_sweep_switch, write_sweep_temp = get_write_current_array(
        write_sweep_dict_list
    )
    left_branch_current, right_branch_current = calculate_branch_currents_read(
        read_current, ALPHA
    )

    # Plot
    plot_write_operation_figure()

    plot_sub_figures()
