import collections
import os
from typing import List, Literal, Tuple

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import nmem.analysis.plot_config
from nmem.analysis.analysis import (
    get_read_current,
    get_state_current_markers,
    get_write_current,
    import_directory,
    plot_read_sweep_array,
    plot_read_switch_probability_array,
    plot_fill_between_array
)
from nmem.measurement.cells import CELLS

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


CRITICAL_CURRENT_ZERO = 1600
ALPHA = 0.563
RETRAP = 0.573
WIDTH = 1 / 2.13


if __name__ == "__main__":
    colors = {0: "blue", 1: "blue", 2: "red", 3: "red"}
    fig, axs = plt.subplot_mosaic("AA;CD", figsize=(8.3, 4))
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\write_current_sweep_enable_write\data"
    )
    # fig, ax = plt.subplots(figsize=(6, 4))
    dict_list = dict_list[::5]
    dict_list = dict_list[::-1]
    # ax, ax2 = plot_write_sweep(axs["B"], dict_list)
    read_current = get_read_current(dict_list[0])
    # ax.set_xlim(0, 300)
    # ax.legend(
    #     frameon=False,
    #     loc="upper left",
    #     bbox_to_anchor=(1.1, 1),
    #     title="Enable Write Temperature [K]",
    # )

    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\read_current_sweep_write_current2\write_current_sweep_C3"
    )
    ax = plot_read_sweep_array(
        axs["A"],
        dict_list,
        "bit_error_rate",
        "write_current",
    )
    plot_read_switch_probability_array(ax, dict_list)
    # plot_fill_between_array(ax, dict_list)
    ax.axvline(read_current, color="black", linestyle="-.")
    write_current_fixed = 100
    ax.set_xlabel("$I_{\mathrm{read}}$ ($\mu$A)")
    ax.set_ylabel("BER")
    ax.set_xlim(650, 800)
    ax2 = ax.twinx()
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Switching Probability")
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))

    # axs["B"].axvline(write_current_fixed, color="black", linestyle="--")
    fig.subplots_adjust(wspace=0.2, hspace=0.3)

    # plt.savefig("write_current_sweep_enable_write2.pdf", bbox_inches="tight")

    # fig, axs = plt.subplot_mosaic("CD", figsize=(7.5, 3))

    ax = axs["D"]
    for data_dict in dict_list:
        # plot_read_sweep(ax, data_dict, "bit_error_rate", "read_current")
        state_current_markers = get_state_current_markers(data_dict, "read_current")
        write_current = get_write_current(data_dict)
        for i, state_current in enumerate(state_current_markers[0, :]):
            if state_current > 0:
                ax.plot(
                    write_current,
                    state_current,
                    "o",
                    label=f"{write_current} $\mu$A",
                    markerfacecolor=colors[i],
                    markeredgecolor="none",
                )
        ax.axhline(read_current, color="black", linestyle="-.")
        # plot_state_current_markers(ax, data_dict, "read_current")
    # ax.axvline(write_current_fixed, color="black", linestyle="--")
    ax.set_xlim(0, 300)
    ax.set_ylabel("State Current [$\mu$A]")
    ax.set_xlabel("Write Current [$\mu$A]")

    ax = axs["C"]
    for data_dict in dict_list:
        state_current_markers = get_state_current_markers(data_dict, "read_current")
        write_current = get_write_current(data_dict)
        for i, state_current in enumerate(state_current_markers[1, :]):
            if state_current > 0:
                ax.plot(
                    write_current,
                    state_current,
                    "o",
                    label=f"{write_current} $\mu$A",
                    markerfacecolor=colors[i],
                    markeredgecolor="none",
                )

    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    # ax.axvline(write_current_fixed, color="black", linestyle="--")
    ax.set_ylabel("Bit Error Rate")
    ax.set_xlabel("$I_{\mathrm{write}}$ ($\mu$A)")

    plt.savefig("read_current_sweep_operating.pdf", bbox_inches="tight")
