import matplotlib.pyplot as plt
import numpy as np

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
    CRITICAL_TEMP,
)

ALPHA = 0.545
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
        if j == 0:
            read_temperature = np.array([get_channel_temperature(data_dict, "read")])
            critical_current_zero = get_critical_current_intercept(data_dict)
            channel_current_zero = calculate_critical_current_temp(
                read_temperature,
                CRITICAL_TEMP,
                critical_current_zero,
            )

        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_current = get_write_current(data_dict)
        read_current = get_read_current(data_dict)

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


if __name__ == "__main__":
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )
    ichl, irhl, ichr, irhr = get_branch_currents(dict_list[0])

    fig, axs = plt.subplots(1, 2, figsize=(6, 4), constrained_layout=True)
    ax = axs[0]

    persistent_current = np.linspace(0, 100, 1000)
    persistent_current = np.where(persistent_current > irhl, irhl, persistent_current)
    read_current = get_read_current(dict_list[0])
    
    left_branch_current = read_current*ALPHA
    right_branch_current = read_current*(1-ALPHA)
    (
        write_current_array,
        write_temp_array,
        write_channel_array,
        write_critical_currents,
    ) = process_dict_list(dict_list)

    for i in range(4):
        ax.plot(
            write_current_array,
            write_channel_array[:, i],
            linestyle="-",
            marker="o",
            color=RBCOLORS[i],
        )
    limits = ax.get_ylim()
    ic_limits = calculate_critical_current_temp(
        np.array(limits), CRITICAL_TEMP, get_critical_current_intercept(dict_list[0])
    )

    ax.set_ylabel("$I_{\mathrm{CH}}$ [$\mu$A]")
    ax.set_xlim(0, IWRITE_XLIM)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid(axis="x")
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")

    ax.axhline(ichl + irhr, color="grey", linestyle="--", label="ichl + irhr")
    ax.axhline(ichr, color="red", linestyle="--", label="ichr")
    ax.axhline(ichr + irhl, color="black", linestyle="--", label="ichr + irhl")
    ax.axhline(
        read_current * ALPHA, color="green", linestyle="--", label="i_L"
    )
    ax.axhline(
        read_current * (1 - ALPHA),
        color="green",
        linestyle="--",
        label="i_R",
    )
    ax.set_aspect(1 / ax.get_data_ratio())
    ax.legend()


    ax.fill_between(
        np.linspace(0, IWRITE_XLIM, 1000),
        left_branch_current-persistent_current,
        left_branch_current+persistent_current,
        color="green",
        alpha=0.1,
    )
    ax.fill_between(
        np.linspace(0, IWRITE_XLIM, 1000),
        right_branch_current-persistent_current,
        right_branch_current+persistent_current,
        color="blue",
        alpha=0.1,
    )


    ax = axs[1]
    ax.plot(
        np.linspace(0, IWRITE_XLIM, 1000),
        persistent_current,
        color="black",
        linestyle="-",
    )
    ax.set_xlim(0, IWRITE_XLIM)
    ax.set_ylim(0, MAX_PERSISTENT)
    ax.set_aspect(1 / ax.get_data_ratio())
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$I_{\mathrm{P}}$ [$\mu$A]")    
    ax.axvline(irhl, color="black", linestyle="--")


    # fig, ax = plt.subplots()
