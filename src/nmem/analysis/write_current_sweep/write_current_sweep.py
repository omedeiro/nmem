import os

import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    build_array,
    filter_first,
    import_directory,
    plot_write_sweep, 
    plot_read_temp_sweep_C3
)
from nmem.measurement.functions import calculate_channel_temperature, calculate_critical_current
from matplotlib.ticker import MultipleLocator



if __name__ == "__main__":
    # plot_write_sweep("write_current_sweep_B2_0")
    # plot_write_sweep("write_current_sweep_B2_1")
    # plot_write_sweep("write_current_sweep_B2_2")

    fig, ax = plt.subplots()
    plot_write_sweep(ax, "write_current_sweep_A2")
    plt.show()

    fig, ax = plt.subplots()
    plot_write_sweep(ax, "write_current_sweep_C2")
    plt.show()



    plot_read_temp_sweep_C3()