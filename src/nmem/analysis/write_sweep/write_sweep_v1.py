import matplotlib.pyplot as plt
import scipy.io as sio

from nmem.analysis.analysis import (
    plot_bit_error_rate,
    plot_voltage_trace_stack,
    import_directory,
)

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 6


TRACE_INDEX = 10

if __name__ == "__main__":
    dict_list = import_directory("data")
    data_off = dict_list[0]

    data_on = dict_list[1]


    # Plots with the enable on
    fig, ax = plt.subplots()
    ax = plot_bit_error_rate(ax, data_on)
    ax.set_xlabel("Write Current [$\mu$A]")

    fig, axs = plt.subplots(3, 1)
    ax = plot_voltage_trace_stack(axs, data_on, trace_index=TRACE_INDEX)

    # Plots with the enable off
    fig, ax = plt.subplots()
    ax = plot_bit_error_rate(ax, data_off)
    ax.set_xlabel("Write Current [$\mu$A]")

    fig, ax = plt.subplots(3, 1)
    ax = plot_voltage_trace_stack(ax, data_off, trace_index=TRACE_INDEX)
