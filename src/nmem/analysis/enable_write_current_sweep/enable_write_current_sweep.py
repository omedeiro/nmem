
import matplotlib.pyplot as plt
import scipy.io as sio

from nmem.analysis.analysis import (
    plot_enable_write_sweep_multiple,
    # plot_peak_distance,
    plot_operating_points,
    plot_operating_margins,
    plot_waterfall,
    import_directory,
)

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 12



if __name__ == "__main__":

    data_list = import_directory("data")

    fig, ax = plt.subplots()
    plot_enable_write_sweep_multiple(ax, data_list)
    plt.show()

    fig, ax = plt.subplots()
    plot_operating_points(ax, data_list)
    plt.show()

    fig, ax = plt.subplots()
    plot_operating_margins(ax, data_list)
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
    plot_waterfall(ax, data_list)
    plt.show()