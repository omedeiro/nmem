import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.axes import Axes

from nmem.analysis.analysis import (
    plot_trace_averaged,
    plot_hist,
    plot_bitstream,
    plot_delay,
    import_directory,
    create_trace_hist_plot,
)


plt.rcParams["figure.figsize"] = [7, 3.5]
plt.rcParams["font.size"] = 5
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.frameon"] = False


plt.rcParams["xtick.major.size"] = 1
plt.rcParams["ytick.major.size"] = 1


if __name__ == "__main__":

    dict_list = import_directory("data")
    fig, ax = plt.subplots()
    plot_delay(ax, dict_list)
    plt.show()

    fig, ax = plt.subplots()
    plot_hist(ax, dict_list[3])
    plt.show()


    dict_list2 = import_directory("data2")
    for data_dict in dict_list2:
        fig, ax = plt.subplots()
        plot_bitstream(ax, data_dict, "trace_chan_out")
        plt.show()


    fig = plt.figure(figsize=(6.264, 2))
    ax_dict = fig.subplot_mosaic("AC;BC")
    create_trace_hist_plot(ax_dict, dict_list)
    fig.subplots_adjust(wspace=0.45)
    plt.show()
    save = False
    if save:
        plt.savefig("delay_plotting_v2.pdf", bbox_inches="tight")

