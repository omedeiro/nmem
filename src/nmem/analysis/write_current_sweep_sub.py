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

WIDTH = 0.35
RETRAP = 0.2
IWRITE_XLIM = 100
IWRITE_XLIM_2 = 300
MAX_PERSISTENT=50
RBCOLORS = {0: "black", 1: "black", 2: "grey", 3: "grey"}
if __name__ == "__main__":
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )

    fig, ax = plt.subplots()

    persistent_current = np.linspace(0, 100, 1000)
    persistent_current = np.where(persistent_current > MAX_PERSISTENT, MAX_PERSISTENT, persistent_current)
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
            ichl, irhl, ichr, irhr = calculate_branch_currents(
                read_temperature, CRITICAL_TEMP, RETRAP, WIDTH, channel_current_zero
            )
            print(f"ichl: {ichl},\n irhl: {irhl},\n ichr: {ichr},\n irhr: {irhr}")

        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_current = get_write_current(data_dict)
        write_critical_currents = calculate_critical_current_temp(
            np.array([get_channel_temperature(data_dict, "write")]),
            CRITICAL_TEMP,
            critical_current_zero,
        )
        write_temps = get_channel_temperature_sweep(data_dict)
        write_current_array[j] = write_current
        critical_current_zero = get_critical_current_heater_off(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                write_temp_array[j, i] = write_temps[arg]
                write_channel_array[j, i] = calculate_critical_current_temp(
                    write_temps[arg],
                    CRITICAL_TEMP,
                    critical_current_zero,
                )
    for i in range(4):
        ax.plot(
            write_current_array,
            write_channel_array[:, i],
            linestyle="--",
            marker="o",
            color=RBCOLORS[i],
        )
    limits = ax.get_ylim()
    ic_limits = calculate_critical_current_temp(
        np.array(limits), CRITICAL_TEMP, critical_current_zero
    )

    ax.set_ylabel("$I_{\mathrm{CH}}$ [$\mu$A]")
    ax.set_xlim(0, IWRITE_XLIM)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid(axis="x")
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")

    ax.axhline(ichl + irhr, color="grey", linestyle="--")
    ax.axhline(ichr, color="red", linestyle="--")
    ax.axhline(ichr + irhl, color="black", linestyle="--")

    # ax2 = ax.twinx()
    # ax2.plot(np.linspace(0, IWRITE_XLIM, 1000), persistent_current, color="black", linestyle="-")