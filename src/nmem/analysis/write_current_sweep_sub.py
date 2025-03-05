import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    CRITICAL_TEMP,
    calculate_branch_currents,
    calculate_critical_current_temp,
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_channel_temperature,
    get_channel_temperature_sweep,
    get_critical_current_heater_off,
    get_critical_current_intercept,
    get_read_current,
    get_write_current,
    import_directory,
)

ALPHA = 0.54
WIDTH = 0.35
RETRAP = 0.26
IWRITE_XLIM = 100
IWRITE_XLIM_2 = 300
MAX_PERSISTENT = 50
IREAD = 727.5
RBCOLORS = {0: "black", 1: "black", 2: "grey", 3: "grey"}


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
    ax.set_ylabel("$I_{\mathrm{CH}}$ [$\mu$A]")
    ax.grid(axis="x")
    ax.set_aspect(1 / ax.get_data_ratio())
    return ax


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
    write_currents = np.linspace(0, IWRITE_XLIM, 1000)
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


def plot_persistent_current_est(ax: plt.Axes, persistent_current, irhl):
    ax.plot(
        np.linspace(0, IWRITE_XLIM, 1000),
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


def calculate_persistent_currents(
    write_currents: np.ndarray, left_retrapping_current: float
) -> np.ndarray:
    persistent_currents = np.where(
        write_currents > left_retrapping_current / 2,
        np.abs(write_currents - left_retrapping_current),
        write_currents,
    )
    persistent_currents = np.where(
        persistent_currents > left_retrapping_current,
        left_retrapping_current,
        persistent_currents,
    )

    return persistent_currents


def calculate_branch_currents_read(read_currents, alpha):
    left_branch_current = read_currents * alpha
    right_branch_current = read_currents * (1 - alpha)
    return left_branch_current, right_branch_current


if __name__ == "__main__":
    # Import
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )

    # Preprocess
    read_current = get_read_current(dict_list[0])
    ichl, irhl, ichr, irhr = get_branch_currents(dict_list[0])
    persistent_currents = calculate_persistent_currents(
        np.linspace(0, IWRITE_XLIM, 1000), irhl
    )
    left_branch_current, right_branch_current = calculate_branch_currents_read(
        read_current, ALPHA
    )

    (
        write_current_array,
        write_temp_array,
        write_channel_array,
        write_critical_currents,
    ) = process_dict_list(dict_list)

    ic_limits = calculate_critical_current_temp(
        np.array([write_current_array[0, 0], write_current_array[0, -1]]),
        CRITICAL_TEMP,
        get_critical_current_intercept(dict_list[0]),
    )

    # Plot
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
