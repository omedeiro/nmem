import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.core_analysis import (
    get_channel_temperature,
    get_enable_read_current,
    get_enable_write_current,
)
from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    CMAP,
    plot_enable_read_sweep,
    plot_enable_read_temp,
    plot_enable_write_sweep,
    plot_enable_write_temp,
    set_plot_style,
)

set_plot_style()






if __name__ == "__main__":
    # Import
    data = import_directory("data")
    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    enable_read_310_C4_list = import_directory("data_310uA_C4")
    data_inverse = import_directory("data_inverse")

    dict_list = [enable_read_290_list, enable_read_300_list, enable_read_310_list]
    dict_list = dict_list[2]

    data_list = import_directory("../read_current_sweep_enable_write/data")
    data_list2 = [data_list[0], data_list[3], data_list[-6], data_list[-1]]
    colors = CMAP(np.linspace(0, 1, len(data_list2)))

    # Preprocess
    read_temperatures = []
    enable_read_currents = []
    for data_dict in dict_list:
        read_temperature = get_channel_temperature(data_dict, "read")
        enable_read_current = get_enable_read_current(data_dict)
        read_temperatures.append(read_temperature)
        enable_read_currents.append(enable_read_current)

    enable_write_currents = []
    write_temperatures = []
    for i, data_dict in enumerate(data_list):
        enable_write_current = get_enable_write_current(data_dict)
        write_temperature = get_channel_temperature(data_dict, "write")
        enable_write_currents.append(enable_write_current)
        write_temperatures.append(write_temperature)

    # Plot
    fig, axs = plt.subplots(
        2, 2, figsize=(6, 3), constrained_layout=True, width_ratios=[1, 0.25]
    )

    ax: plt.Axes = axs[1, 0]
    plot_enable_read_sweep(ax, dict_list[::-1], marker=".")

    ax: plt.Axes = axs[1, 1]
    plot_enable_read_temp(ax, enable_read_currents, read_temperatures)

    ax = axs[0, 0]
    plot_enable_write_sweep(ax, data_list2, marker=".")

    ax = axs[0, 1]
    plot_enable_write_temp(ax, enable_write_currents, write_temperatures)

    save = False
    if save:
        fig.savefig("read_current_sweep_enable_read.pdf", bbox_inches="tight")
