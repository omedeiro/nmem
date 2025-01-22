import os

import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    build_array,
    filter_first,
    import_directory,
    plot_write_sweep,
    plot_channel_temperature,
    plot_read_temp_sweep_C3
)
from nmem.measurement.functions import calculate_channel_temperature, calculate_critical_current, CELLS
from matplotlib.ticker import MultipleLocator
SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3

    
if __name__ == "__main__":

    fig, ax = plt.subplots()
    plot_write_sweep(ax, r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\write_sweep_C3")
    plt.show()

    # fig, ax = plt.subplots()
    # plot_write_sweep(ax, r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\write_sweep_C3")
    # plt.show()

    data_list = import_directory(
         r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\write_sweep_C3"
    )
    fig, ax = plt.subplots()
    for data_dict in data_list:
        plot_channel_temperature(ax,  data_dict, marker="o", color="b")
    plt.show()

    # plot_read_temp_sweep_C3()