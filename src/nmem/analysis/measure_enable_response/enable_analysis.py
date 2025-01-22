import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from nmem.analysis.analysis import plot_all_cells
from nmem.measurement.cells import CELLS


if __name__ == "__main__":
    # plot_full_grid()
    # plot_all_cells()

    # dict_list = import_directory("data")
    # fig, ax = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    # plot_column(ax, dict_list)
    # plt.show()

    # dict_list = import_directory("data")
    # fig, axs = plt.subplots(4,4, figsize=(20, 20), sharex=True, sharey=True)
    # plot_grid(axs, dict_list)

    fig, ax = plt.subplots()
    plot_all_cells(ax)
