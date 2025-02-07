import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_enable_write_current,
    get_read_currents,
    get_write_temperature,
    import_directory,
    plot_write_sweep,
)

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3



if __name__ == "__main__":
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\write_current_sweep_enable_write\data"
    )
    dict_list = dict_list[1:]
    fig, axs = plt.subplots(1, 2, figsize=(6, 4), width_ratios=[1, 0.25])
    dict_list = dict_list[::-1]
    ax = axs[0]
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
        write_currents = get_read_currents(data_dict)
        enable_write_current = get_enable_write_current(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:

                if i==0:
                    ichl_current_list.append(write_currents[arg])
                    ichl_temp.append(get_write_temperature(data_dict))
                    # ax.plot(
                    #     write_currents[arg],
                    #     bit_error_rate[arg],
                    #     color="C0",
                    #     marker="o",
                    #     markersize=5,
                    # )
                if i==2:
                    ichr_current_list.append(write_currents[arg])
                    ichr_temp.append(get_write_temperature(data_dict))
                    # ax.plot(
                    #     write_currents[arg],
                    #     bit_error_rate[arg],
                    #     color="C1",
                    #     marker="o",
                    #     markersize=5,
                    # )                    
    ax.axvline(30, color="grey", linestyle="--")
    ax.axvline(110, color="grey", linestyle="--")
    ax.axvline(140, color="grey", linestyle="--")

    ax = axs[1]


    ax.plot(ichl_temp, ichl_current_list, marker="o")
    ax.plot(ichr_temp, ichr_current_list, marker="o")
    ax.set_xlabel("$T_{\mathrm{write}}$ [K]")
    ax.set_ylabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylim(0,300)
    ax.grid()


    fig.subplots_adjust(wspace=0.3)