import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from nmem.analysis.analysis import (
    build_array,
    get_fitting_points,
    plot_fit,
    convert_cell_to_coordinates,
    import_directory,
    get_current_cell,
    plot_enable_current_relation,
)

plt.rcParams["figure.figsize"] = [5.7, 5]
plt.rcParams["font.size"] = 16


if __name__ == "__main__":
    dict_list = import_directory("data")
    fig, ax = plt.subplots()
    plot_enable_current_relation(ax, dict_list)
    plt.show()