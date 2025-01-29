import matplotlib.pyplot as plt
import scipy.io as sio
import os

from nmem.analysis.analysis import plot_read_sweep_array, import_directory

plt.rcParams["figure.figsize"] = [5, 3.5]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    dict_list = import_directory("data")
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "write_width")
    plt.show()
