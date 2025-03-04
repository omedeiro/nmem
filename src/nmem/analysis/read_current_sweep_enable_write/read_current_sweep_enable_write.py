import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    CMAP,
    get_channel_temperature,
    get_enable_write_current,
    import_directory,
)
from nmem.analysis.read_current_sweep_enable_read.read_current_sweep_enable_read import (
    plot_enable_write_sweep,
    plot_enable_write_temp,
)


def get_enable_write_temp(dict_list):
    enable_write_currents = []
    write_temperatures = []

    for data_dict in dict_list:
        enable_write_current = get_enable_write_current(data_dict)

        write_temperature = get_channel_temperature(data_dict, "write")
        enable_write_currents.append(enable_write_current)
        write_temperatures.append(write_temperature)

    return enable_write_currents, write_temperatures


colors = CMAP(np.linspace(0, 1, 4))


if __name__ == "__main__":
    # Import
    data_list = import_directory("data")
    data_list2 = [data_list[0], data_list[3], data_list[-6], data_list[-1]]


    # Plot
    fig, axs = plt.subplots(
        1, 2, figsize=(8.37, 2), constrained_layout=True, width_ratios=[1, 0.25]
    )

    ax = axs[0]
    plot_enable_write_sweep(ax, data_list2, colors)

    ax = axs[1]
    enable_write_currents, write_temperatures = get_enable_write_temp(data_list)
    plot_enable_write_temp(ax, enable_write_currents, write_temperatures, colors)

    save = False
    if save:
        fig.savefig("read_current_sweep_enable_write2.pdf", bbox_inches="tight")
    plt.show()
