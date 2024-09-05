import matplotlib.pyplot as plt
import scipy.io as sio

from nmem.analysis.analysis import plot_measurement_coarse, plot_trace_stack_write

plt.rcParams["figure.figsize"] = [6, 2]
plt.rcParams["font.size"] = 12


TRACE_INDEX = 10

if __name__ == "__main__":
    data_on = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 16-06-54.mat"
    )

    data_off = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 16-14-55.mat"
    )

    zoom_write = sio.loadmat(
        "SPG806_20240830_nMem_optimize_read_D6_A4_C1_2024-08-30 09-50-30.mat"
    )

    fig, ax = plt.subplots()
    ax = plot_measurement_coarse(ax, data_on)
    plt.show()

    fig, ax = plt.subplots()
    ax = plot_trace_stack_write(ax, data_on, TRACE_INDEX)
    plt.show()

    fig, ax = plt.subplots()
    ax = plot_measurement_coarse(ax, data_off)
    plt.show()

    fig, ax = plt.subplots()
    ax = plot_trace_stack_write(ax, data_off, TRACE_INDEX)
    plt.show()
