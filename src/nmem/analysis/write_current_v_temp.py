import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    calculate_critical_current_temp,
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_channel_temperature_sweep,
    get_write_current,
    import_directory,
)

if __name__ == "__main__":
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )

    fig, axs = plt.subplot_mosaic("A")

    ax = axs["A"]
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
    ax.set_ylim([8.3, 9.7])
    ax2 = ax.twinx()

    colors = {
        0: "blue",
        1: "blue",
        2: "red",
        3: "red",
    }
    for data_dict in dict_list:
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_current = get_write_current(data_dict)
        write_temps = get_channel_temperature_sweep(data_dict)

        for i, arg in enumerate(berargs):
            write_temp = write_temps[arg]
            ax.plot(
                write_current,
                write_temp,
                "o",
                label=f"{write_current} $\mu$A",
                markerfacecolor=colors[i],
                markeredgecolor="none",
                markersize=3,
            )
            ic = calculate_critical_current_temp(write_temp, 12.3, 910)
            limits = ax.get_ylim()
            ic_limits = calculate_critical_current_temp(np.array(limits), 12.3, 910)
            ax2.set_ylim([ic_limits[0], ic_limits[1]])

    ax2.set_ylabel("$I_{\mathrm{CH}}$ [$\mu$A]")
    ax.set_xlim(0, 100)
    # ax.set_ylim(200, 400)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.grid()
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$T_{\mathrm{write}}$ [$\mu$A]")
