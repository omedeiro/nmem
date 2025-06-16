import matplotlib.pyplot as plt

from nmem.analysis.constants import TRACE_INDEX
from nmem.analysis.data_import import (
    import_directory,
)
from nmem.analysis.plotting import (
    plot_bit_error_rate,
)
from nmem.analysis.trace_plots import (
    plot_voltage_trace_stack,
)


def main():
    dict_list = import_directory("data")
    data_off = dict_list[0]
    data_on = dict_list[1]

    # Plots with the enable on
    fig, ax = plt.subplots()
    ax = plot_bit_error_rate(ax, data_on)
    ax.set_xlabel("Write Current [$\\mu$A]")

    fig, axs = plt.subplots(3, 1)
    ax = plot_voltage_trace_stack(axs, data_on, trace_index=TRACE_INDEX)

    # Plots with the enable off
    fig, ax = plt.subplots()
    ax = plot_bit_error_rate(ax, data_off)
    ax.set_xlabel("Write Current [$\\mu$A]")

    fig, axs = plt.subplots(3, 1)
    ax = plot_voltage_trace_stack(axs, data_off, trace_index=TRACE_INDEX)


if __name__ == "__main__":
    main()
