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
    CRITICAL_TEMP
)

if __name__ == "__main__":
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )

    fig, axs = plt.subplot_mosaic(
        "AB;CD", figsize=(8.3, 4), width_ratios=[1, 0.25], constrained_layout=True
    )

    ax = axs["A"]
    ax, ax2 = plot_enable_write_sweep_multiple(ax, dict_list[0:6])
    ax.set_ylabel("BER")
    ax2.set_xlabel("$T_{\mathrm{write}}$ [K]")
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")

    ax = axs["B"]
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.20))
    ax.set_ylim([8.3, 9.7])
    ax2 = ax.twinx()
    
    colors = {
        0: "blue",
        1: "blue",
        2: "red",
        3: "red",
    }
    write_temp_array = np.empty((len(dict_list), 4))
    write_current_array = np.empty((len(dict_list), 1))
    for j, data_dict in enumerate(dict_list):
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_current = get_write_current(data_dict)
        write_temps = get_channel_temperature_sweep(data_dict)
        write_current_array[j] = write_current
        critical_current_zero = get_critical_current_heater_off(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                write_temp_array[j, i] = write_temps[arg]
            # write_temp = write_temps[arg]
    for i in range(4):
        ax.plot(
            write_current_array,
            write_temp_array[:, i],
            linestyle="--",
            marker="o",
            color=colors[i],
            # label="Write 0",
        )
    limits = ax.get_ylim()
    ic_limits = calculate_critical_current_temp(np.array(limits), CRITICAL_TEMP, critical_current_zero)
    ax2.set_ylim([ic_limits[0], ic_limits[1]])

    ax2.set_ylabel("$I_{\mathrm{CH}}$ [$\mu$A]")
    ax.set_xlim(0, 100)
    # ax.set_ylim(200, 400)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid()
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$T_{\mathrm{write}}$ [K]")

    ax = axs["C"]
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\write_current_sweep_enable_write\data"
    )
    dict_list = dict_list[1:]
    dict_list = dict_list[::-1]
    plot_write_sweep(ax, dict_list)
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(0, 300)
    ichl_current_list = []
    ichr_current_list = []
    ichl_temp = []
    ichr_temp = []
    for data_dict in dict_list:
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_currents = get_read_currents(
            data_dict
        )  # This is correct. "y" is the write current in this .mat.    
        enable_write_current = get_enable_write_current(data_dict)
        read_current = get_read_current(data_dict)
        write_current = get_write_current(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:

                if i == 0:
                    ichl_current_list.append(write_currents[arg])
                    ichl_temp.append(get_channel_temperature(data_dict, "write"))
                    # ax.plot(
                    #     write_currents[arg],
                    #     bit_error_rate[arg],
                    #     color="C0",
                    #     marker="o",
                    #     markersize=5,
                    # )
                if i == 2:
                    ichr_current_list.append(write_currents[arg])
                    ichr_temp.append(get_channel_temperature(data_dict, "write"))
                    # ax.plot(
                    #     write_currents[arg],
                    #     bit_error_rate[arg],
                    #     color="C1",
                    #     marker="o",
                    #     markersize=5,
                    # )
    # ax.axvline(30, color="grey", linestyle="--")
    # ax.axvline(110, color="grey", linestyle="--")
    # ax.axvline(140, color="grey", linestyle="--")
    ax = axs["D"]

    ax.plot(ichl_temp, ichl_current_list, linestyle="--", marker="o", color="blue")
    ax.plot(ichr_temp, ichr_current_list, linestyle="--", marker="o", color="red")
    ax.set_xlabel("$T_{\mathrm{write}}$ [K]")
    ax.set_ylabel("$I_{\mathrm{ch}}$ [$\mu$A]")
    ax.set_ylim(0, 300)
    ax.grid()

    fig.savefig("write_current_sweep_operation.pdf", bbox_inches="tight")
