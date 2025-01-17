import os

import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    build_array,
    filter_first,
    import_directory,
)
from nmem.measurement.functions import calculate_channel_temperature, calculate_critical_current
from matplotlib.ticker import MultipleLocator
SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3

def plot_write_sweep(ax, data_directory: str):
    data_list = import_directory(data_directory)
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_list)))

    ax2 = ax.twinx()

    for data in data_list:
        x, y, ztotal = build_array(data, "bit_error_rate")
        _, _, zswitch = build_array(data, "total_switches_norm")
        ax.plot(
            y,
            ztotal,
            label=f"write current = {filter_first(data.get('write_current'))*1e6} $\mu$A",
            color=colors[data_list.index(data)],
        )
        ax2.plot(
            y,
            zswitch,
            label="_",
            color="grey",
            linewidth=0.5,
            linestyle=":",
        )

        cell = data.get("cell")[0]
        max_enable_current = filter_first(data["CELLS"][cell][0][0]["x_intercept"]).flatten()[0]
        enable_read_current = filter_first(data.get("enable_read_current"))*1e6
        enable_write_current = filter_first(data.get("enable_write_current"))*1e6

        
        enable_read_temp = calculate_channel_temperature(
            SUBSTRATE_TEMP,
            CRITICAL_TEMP,
            enable_read_current,
            max_enable_current
        ).flatten()[0]
        enable_write_temp = calculate_channel_temperature(
            SUBSTRATE_TEMP,
            CRITICAL_TEMP,
            enable_write_current,
            max_enable_current
        ).flatten()[0]

    ax.set_title(
        f"Cell {cell} - Read Temperature: {enable_read_temp:.2f}, Write Temperature: {enable_write_temp:.2f} K"
    )
    ax.legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")
    ax.set_ylim([0, 1])
    ax.set_xlim([500, 900])
    ax.set_xlabel("Read Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Switching Probability")
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    # calculate_critical_current(enable_read_current, data["CELLS"][cell][0][0])

def plot_read_temp_sweep_C3():
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    plot_write_sweep(axs[0,0], "write_current_sweep_C3_2")
    plot_write_sweep(axs[0,1], "write_current_sweep_C3_3")
    plot_write_sweep(axs[1,0], "write_current_sweep_C3_4")
    plot_write_sweep(axs[1,1], "write_current_sweep_C3")
    axs[1,1].legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
if __name__ == "__main__":
    # plot_write_sweep("write_current_sweep_B2_0")
    # plot_write_sweep("write_current_sweep_B2_1")
    # plot_write_sweep("write_current_sweep_B2_2")

    fig, ax = plt.subplots()
    plot_write_sweep(ax, "write_current_sweep_A2")
    plt.show()

    fig, ax = plt.subplots()
    plot_write_sweep(ax, "write_current_sweep_C2")
    plt.show()



    plot_read_temp_sweep_C3()