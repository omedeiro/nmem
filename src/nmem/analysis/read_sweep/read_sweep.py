import matplotlib.pyplot as plt
import scipy.io as sio
from nmem.analysis.analysis import (
    plot_hist,
    plot_measurement,
    plot_trace_stack_1D,
)

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14

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
    plot_histogram_measurement()
    plot_data_baseline()
    plot_data_delay()
    plot_data_emulate()
