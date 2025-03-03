import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from nmem.analysis.analysis import (
    calculate_critical_current_temp,
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_enable_write_current,
    get_read_currents,
    get_write_current,
    import_directory,
    plot_enable_write_sweep_multiple,
    plot_write_sweep,
    get_read_current,
    get_channel_temperature_sweep,
    get_channel_temperature,
    get_critical_current_heater_off,
    get_critical_current_intercept,
    calculate_branch_currents,
    get_enable_current_sweep,
    CRITICAL_TEMP,
    WIDTH,
    RETRAP,
    ALPHA,
)

IWRITE_XLIM = 100
IWRITE_XLIM_2 = 300

RBCOLORS = {0: "black", 1: "black", 2: "grey", 3: "grey"}


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


def get_branch_currents(data_dict):
    read_temperature = np.array([get_channel_temperature(data_dict, "read")])
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
    ax.set_ylabel("$I_{\mathrm{enable}}$ [$\mu$A]")
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
    IRHL_TR = 47 * 2

    (
        write_current_array,
        write_temp_array,
        write_channel_array,
        critical_current_zero,
    ) = process_dict_list(enable_sweep_dict_list)

    read_current = get_read_current(enable_sweep_dict_list[0])
    ichl, irhl, ichr, irhr = get_branch_currents(enable_sweep_dict_list[0])

    write_sweep_markers, write_temperature_markers = get_write_current_array(
        enable_sweep_dict_list
    )

    write_currents = np.linspace(0, IWRITE_XLIM_2, 1000)
    persistent_current = np.where(
        write_currents > IRHL_TR / 2,
        np.abs(write_currents - IRHL_TR),
        write_currents,
    )
    persistent_current = np.where(
        persistent_current > IRHL_TR, IRHL_TR, persistent_current
    )


    write_sweep_switch, write_sweep_temp = get_write_current_array(write_sweep_dict_list)

    # Plot
    fig, axs = plt.subplot_mosaic(
        "AB;CD", figsize=(8.3, 4), width_ratios=[1, 0.25], constrained_layout=True
    )

    ax = axs["A"]
    ax, ax2 = plot_enable_write_sweep_multiple(ax, enable_sweep_dict_list[0:6])
    ax.set_ylabel("BER")
    ax2.set_xlabel("$T_{\mathrm{write}}$ [K]")
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    ax.grid()

    ax = axs["B"]
    for i in range(4):
        ax = plot_write_array(
            ax, write_current_array, write_channel_array[:, i], color=RBCOLORS[i]
        )

    # Write current sweep
    ax = axs["C"]
    ax = plot_write_sweep(ax, write_sweep_dict_list)
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(0, IWRITE_XLIM_2)

    ax.axvline(irhl, color="C0", linestyle="--", label="_irhl")
    ax.axvline(irhr, color="C1", linestyle="--", label="_irhr")
    ax.axvline(ichl, color="C2", linestyle="--", label="_ichl")
    ax.axvline(ichr, color="C3", linestyle="--", label="_ichr")

    ax.plot(write_currents, persistent_current / IRHL_TR, color="black", linestyle="--")

    ax = axs["D"]
    ax = plot_write_current_sweep_markers(
        ax, write_sweep_temp[:, 0], write_sweep_switch[:, 0], color="black"
    )
    ax = plot_write_current_sweep_markers(
        ax, write_sweep_temp[:, 2], write_sweep_switch[:, 2], color="grey"
    )

    ax.axhline(10, color="C2", linestyle="--", label="_ichl")
    ax.axhline(110, color="C2", linestyle="--", label="_ichl")

    fig.savefig("write_current_sweep_operation.pdf", bbox_inches="tight")

    plt.show()