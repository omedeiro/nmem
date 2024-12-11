import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.axes import Axes
from nmem.analysis.analysis import (
    plot_voltage_hist,
    plot_voltage_trace,
    plot_measurement,
    plot_threshold,
    plot_trace_stack_1D,
    text_from_bit,
)

plt.rcParams["figure.figsize"] = [1.77, 3.54]
plt.rcParams["font.size"] = 10


if __name__ == "__main__":
    TRACE_INDEX = 1

    data_delay = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 10-43-20.mat"
    )

    data_hist = sio.loadmat(
        "SPG806_20240831_nMem_parameter_sweep_D6_A4_C1_2024-08-31 13-09-41.mat"
    )

    data_baseline = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 10-46-40.mat"
    )
    data_emulate = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 10-48-39.mat"
    )
    data_emulate_reverse = sio.loadmat(
        "SPG806_20240830_nMem_parameter_sweep_D6_A4_C1_2024-08-30 10-49-06.mat"
    )

    fig, ax = plt.subplots()
    ax = plot_measurement(ax, data_hist)

    fig, ax = plt.subplots()
    read_zero_voltage = data_hist.get("read_zero_top")[0][:, TRACE_INDEX] * 1e3
    read_one_voltage = data_hist.get("read_one_top")[0][:, TRACE_INDEX] * 1e3
    plot_voltage_hist(ax, read_zero_voltage, label="Read 0", color="C0")
    plot_voltage_hist(ax, read_one_voltage, label="Read 1", color="C1")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 4))
    plot_trace_stack_1D(axs, data_baseline)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 4))
    plot_trace_stack_1D(axs, data_delay)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 4))
    plot_trace_stack_1D(axs, data_emulate)