import matplotlib.pyplot as plt
import scipy.io as sio

from nmem.analysis.analysis import (
    plot_bit_error_rate,
    plot_trace_stack_1D,
)

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 6


TRACE_INDEX = 10

if __name__ == "__main__":
    data_off = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 16-06-54.mat"
    )

    data_on = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 16-14-55.mat"
    )

    zoom_write = sio.loadmat(
        "SPG806_20240830_nMem_optimize_read_D6_A4_C1_2024-08-30 09-50-30.mat"
    )

    # Plots with the enable on
    fig, ax = plt.subplots()
    ax = plot_bit_error_rate(ax, data_on)
    ax.set_xlabel("Write Current [$\mu$A]")

    fig, axs = plt.subplots(3, 1)
    ax = plot_trace_stack_1D(axs, data_on, trace_index=TRACE_INDEX)

    # Plots with the enable off
    fig, ax = plt.subplots()
    ax = plot_bit_error_rate(ax, data_off)
    ax.set_xlabel("Write Current [$\mu$A]")

    fig, ax = plt.subplots(3, 1)
    ax = plot_trace_stack_1D(ax, data_off, trace_index=TRACE_INDEX)
