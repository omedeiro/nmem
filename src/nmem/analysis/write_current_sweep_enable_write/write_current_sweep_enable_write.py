import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_channel_temperature,
    get_read_currents,
    import_directory,
    plot_write_sweep,
)


def plot_write_sweep_temps(ax: plt.Axes, dict_list: list[dict]):
    plot_write_sweep(ax, dict_list)
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(0, 300)

    return ax

def process_data(dict_list: list[dict]):
    ichl_current_list = []
    ichr_current_list = []
    ichl_temp = []
    ichr_temp = []
    for data_dict in dict_list:
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_currents = get_read_currents(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                if i==0:
                    ichl_current_list.append(write_currents[arg])
                    ichl_temp.append(get_channel_temperature(data_dict, "write"))

                if i==2:
                    ichr_current_list.append(write_currents[arg])
                    ichr_temp.append(get_channel_temperature(data_dict, "write"))


    return ichl_temp, ichr_temp, ichl_current_list, ichr_current_list

def plot_extracted_markers(ax: plt.Axes, ichl_temp, ichr_temp, ichl_current_list, ichr_current_list):
    ax.plot(ichl_temp, ichl_current_list, marker="o")
    ax.plot(ichr_temp, ichr_current_list, marker="o")
    ax.set_xlabel("$T_{\mathrm{write}}$ [K]")
    ax.set_ylabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylim(0,300)
    ax.grid()
    return ax


if __name__ == "__main__":
    # Import
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\write_current_sweep_enable_write\data"
    )
    dict_list = dict_list[1:]
    dict_list = dict_list[::-1]

    # Preprocess

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(6, 4), width_ratios=[1, 0.25])
    plot_write_sweep_temps(axs[0], dict_list)


    plot_extracted_markers(axs[1], *process_data(dict_list))
    
    fig.subplots_adjust(wspace=0.3)