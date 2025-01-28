
import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_enable_write_sweep_multiple,
    plot_operating_margins,
    # plot_peak_distance,
    plot_operating_points,
    plot_waterfall,
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