import matplotlib.pyplot as plt
import scipy.io as sio
import os
from nmem.analysis.analysis import plot_read_sweep_array, import_directory

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    fig, ax = plt.subplots()
    dict_list = import_directory("data3")
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "write_current")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
