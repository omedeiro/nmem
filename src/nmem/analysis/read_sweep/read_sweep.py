import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.ticker import MultipleLocator

from nmem.analysis.analysis import (
    plot_hist,
    plot_measurement,
    plot_message,
    plot_threshold,
    plot_trace_stack_1D,
    plot_trace_zoom,
    text_from_bit,
)

plt.rcParams["figure.figsize"] = [1.77, 3.54]
plt.rcParams["font.size"] = 10

TRACE_INDEX = 14


def plot_data_baseline():
    data_baseline = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 10-46-40.mat"
    )
    fig, ax = plt.subplots(figsize=(5.7, 5.27))
    ax = plot_trace_stack_1D(ax, data_baseline)
    plt.show()

    return data_baseline


def plot_data_delay():
    data_delay = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 10-43-20.mat"
    )
    fig, ax = plt.subplots()
    ax = plot_trace_stack_1D(ax, data_delay)
    plt.show()
    return data_delay


def plot_data_delay_manu():
    data_delay = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 10-43-20.mat"
    )
    fig, ax = plt.subplots()
    data_dict = data_delay
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.subplot(311)
    x = data_dict["trace_chan_in"][0] * 1e6
    y = data_dict["trace_chan_in"][1] * 1e3
    (p1,) = plt.plot(x, y, color="dimgrey", label="Input")
    plt.xticks(np.linspace(x[0], x[-1], 11), labels=None)
    plt.xlim([x[0], x[-1]])
    ax = plt.gca()
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(data_dict["bitmsg_channel"][0]):
        text = text_from_bit(bit)
        plt.text(
            i + 0.65,
            axheight * 1.1,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.ylim([-150, 1100])
    plt.yticks([0, 500, 1200])
    plt.grid(axis="x", which="both")

    plt.subplot(312)
    x = data_dict["trace_enab"][0] * 1e6
    y = data_dict["trace_enab"][1] * 1e3
    (p2,) = plt.plot(x, y, color="#DBB40C", label="Enable")
    plt.xticks(np.linspace(x[0], x[-1], 11))
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.ylim([-10, 100])
    plt.xlim([x[0], x[-1]])
    plt.yticks([0, 50])
    plt.grid(axis="x", which="both")

    plt.subplot(313)
    x = data_dict["trace_chan_out"][0] * 1e6
    y = data_dict["trace_chan_out"][1] * 1e3

    (p3,) = plt.plot(x, y, color="#08519C", label="Output")
    plt.grid(axis="x", which="both")
    ax = plt.gca()
    ax = plot_threshold(ax, 4, 5, 360)
    ax = plot_threshold(ax, 8, 9, 360)
    plt.xlabel("Time [$\mu$s]")
    plt.ylim([-150, 800])
    plt.xlim([0, 10])
    plt.yticks([0, 500])
    ax = plt.gca()
    ax.tick_params(direction="in")
    plt.sca(ax)
    plt.xticks(np.linspace(x[0], x[-1], 3))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    fig = plt.gcf()
    fig.supylabel("Voltage [mV]", x=-0.10, y=0.5)
    plt.savefig("delay_manu.pdf", bbox_inches="tight")
    plt.show()
    return data_delay


def plot_data_emulate():
    data_emulate = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 10-48-39.mat"
    )
    fig, ax = plt.subplots()
    ax = plot_trace_stack_1D(ax, data_emulate)
    plt.show()
    return data_emulate


def plot_data_emulate_reverse():
    data_emulate_reverse = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 10-49-06.mat"
    )
    fig, ax = plt.subplots()
    ax = plot_trace_stack_1D(ax, data_emulate_reverse)
    plt.show()
    return data_emulate_reverse


def plot_histogram_measurement():
    data_hist = sio.loadmat(
        "SPG806_20240831_nMem_parameter_sweep_D6_A4_C1_2024-08-31 13-09-41.mat"
    )

    fig, ax = plt.subplots()
    ax = plot_measurement(ax, data_hist)
    plot_hist(data_hist, TRACE_INDEX)

    plt.show()


if __name__ == "__main__":
    # plot_histogram_measurement()
    # plot_data_baseline()
    plot_data_delay_manu()
    # plot_data_emulate()
