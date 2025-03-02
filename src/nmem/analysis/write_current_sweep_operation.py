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
    get_enable_current_sweep,
    CRITICAL_TEMP,
    WIDTH,
    RETRAP,
    ALPHA,
)

IWRITE_XLIM = 100
IWRITE_XLIM_2 = 300

RBCOLORS = {0: "black", 1: "black", 2: "grey", 3: "grey"}
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
    ax.grid()

    ax = axs["B"]
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.20))
    ax.set_ylim([8.3, 9.7])
    ax2 = ax.twinx()

    write_temp_array = np.empty((len(dict_list), 4))
    write_current_array = np.empty((len(dict_list), 1))
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
        read_current = get_read_current(data_dict)
        write_current = get_write_current(data_dict)
        write_temps = get_channel_temperature_sweep(data_dict)
        write_current_array[j] = write_current
        critical_current_zero = get_critical_current_heater_off(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                write_temp_array[j, i] = write_temps[arg]
    for i in range(4):
        ax.plot(
            write_current_array,
            write_temp_array[:, i],
            linestyle="--",
            marker="o",
            color=RBCOLORS[i],
        )
    ic_limits = calculate_critical_current_temp(
        np.array(ax.get_ylim()), CRITICAL_TEMP, critical_current_zero
    )
    ax2.set_ylim([ic_limits[0], ic_limits[1]])

    ax2.set_ylabel("$I_{\mathrm{CH}}$ [$\mu$A]")
    ax.set_xlim(0, IWRITE_XLIM)
    ax.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax.grid(axis="x")
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$T_{\mathrm{write}}$ [K]")
    ax2.axhline(ichl + irhr, color="grey", linestyle="-", label="I_{min}")
    ax2.axhline(ichr + irhr, color="grey", linestyle="-", label="I_{max}")
    ax2.axhline(ichr, color="C3", linestyle="--")
    ax2.plot([0], read_current * ALPHA, color="C1", marker="p", markersize=8)
    ax2.plot([0], read_current * (1 - ALPHA), color="C1", marker="p", markersize=8)

    # Write current sweep
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\write_current_sweep_enable_write\data"
    )
    dict_list = dict_list[1:]

    ax = axs["C"]
    plot_write_sweep(ax, dict_list)
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(0, IWRITE_XLIM_2)
    data = []
    data2 = []
    for j, data_dict in enumerate(dict_list):
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)

        # This is correct. "y" is the write current in this .mat.
        write_currents = get_read_currents(data_dict)

        enable_write_current = get_enable_write_current(data_dict)
        read_current = get_read_current(data_dict)
        write_current = get_write_current(data_dict)

        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                if i == 0:
                    data.append(
                        {
                            "write_current": write_currents[arg],
                            "write_temp": get_channel_temperature(data_dict, "write"),
                            "read_current": read_current,
                            "enable_write_current": enable_write_current,
                            "read_temp": get_channel_temperature(data_dict, "read"),
                            "read_channel_current": calculate_critical_current_temp(
                                get_channel_temperature(data_dict, "read"),
                                CRITICAL_TEMP,
                                critical_current_zero,
                            ),
                        }
                    )
                if i == 2:
                    data2.append(
                        {
                            "write_current": write_currents[arg],
                            "write_temp": get_channel_temperature(data_dict, "write"),
                            "read_current": read_current,
                            "enable_write_current": enable_write_current,
                            "read_temp": get_channel_temperature(data_dict, "read"),
                            "read_channel_current": calculate_critical_current_temp(
                                get_channel_temperature(data_dict, "read"),
                                CRITICAL_TEMP,
                                critical_current_zero,
                            ),
                        }
                    )

    ax.axvline(irhl, color="C0", linestyle="--", label="_irhl")
    ax.axvline(irhr, color="C1", linestyle="--", label="_irhr")
    ax.axvline(ichl, color="C2", linestyle="--", label="_ichl")
    ax.axvline(ichr, color="C3", linestyle="--", label="_ichr")

    ax = axs["D"]
    ax.plot(
        [d["write_temp"] for d in data],
        [d["write_current"] for d in data],
        "--o",
        color="black",
    )
    ax.plot(
        [d["write_temp"] for d in data2],
        [d["write_current"] for d in data2],
        "--o",
        color="grey",
    )

    ax.axhline(10, color="C2", linestyle="--", label="_ichl")
    ax.axhline(110, color="C2", linestyle="--", label="_ichl")

    ax.set_xlabel("$T_{\mathrm{write}}$ [K]")
    ax.set_ylabel("$I_{\mathrm{ch}}$ [$\mu$A]")
    ax.set_ylim(0, IWRITE_XLIM_2)
    ax.grid(axis="y")

    fig.savefig("write_current_sweep_operation.pdf", bbox_inches="tight")
