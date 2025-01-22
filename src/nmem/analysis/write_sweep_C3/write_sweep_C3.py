import os

import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_channel_temperature,
    plot_write_sweep,
)

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


if __name__ == "__main__":

    fig, ax = plt.subplots()
    plot_write_sweep(ax, os.getcwd())
    plt.show()

    data_list = import_directory(os.getcwd())
    fig, ax = plt.subplots()
    for data_dict in data_list:
        plot_channel_temperature(ax, data_dict, marker="o", color="b")
    plt.show()

