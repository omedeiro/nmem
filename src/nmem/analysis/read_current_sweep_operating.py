import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from nmem.analysis.analysis import (
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_read_current,
    get_read_currents,
    get_state_current_markers,
    get_write_current,
    import_directory,
    plot_read_sweep_array,
    plot_read_switch_probability_array,
)

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
    ax.axvline(read_current, color="black", linestyle="--", linewidth=1)
    write_current_fixed = 100
    ax.set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-3)
    ax.set_ylabel("BER")
    ax.set_xlim(500, 850)
    ax2 = ax.twinx()
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Switching Probability")
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))

    # axs["B"].axvline(write_current_fixed, color="black", linestyle="--")

    # plt.savefig("write_current_sweep_enable_write2.pdf", bbox_inches="tight")

    # fig, axs = plt.subplot_mosaic("CD", figsize=(7.5, 3))

    ax = axs["C"]
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
                    markersize=4,
                )
    ax.axhline(read_current, color="black", linestyle="--", linewidth=1)
        # plot_state_current_markers(ax, data_dict, "read_current")
    # ax.axvline(write_current_fixed, color="black", linestyle="--")
    ax.set_xlim(0, 300)
    ax.set_ylabel("$I_{\mathrm{state}}$ [$\mu$A]")
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")

    ic_list = []
    write_current_list = []
    ic_list2 = []
    write_current_list2 = []
    for data_dict in dict_list:
        write_current = get_write_current(data_dict)

        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        read_currents = get_read_currents(data_dict)
        if not np.isnan(berargs[0]):
            ic_list.append(read_currents[berargs[0]])
            write_current_list.append(write_current)
        if not np.isnan(berargs[2]):
            ic_list.append(read_currents[berargs[2]])
            write_current_list.append(write_current)

        if not np.isnan(berargs[1]):
            ic_list2.append(read_currents[berargs[1]])
            write_current_list2.append(write_current)
        if not np.isnan(berargs[3]):
            ic_list2.append(read_currents[berargs[3]])
            write_current_list2.append(write_current)

    ax.plot(write_current_list, ic_list, "-", color="grey", linewidth=0.5)
    ax.plot(write_current_list2, ic_list2, "-", color="grey", linewidth=0.5)
    ax.set_xlim(0, 300)
    # ax.set_ylim(0, 1)
    # ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylabel("$I_{\mathrm{read}}$ [$\mu$A]")
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    # set the axes background to transparent
    ax.patch.set_alpha(0.0)
    fig.patch.set_visible(False)
    # ax.plot(write_current_list2, np.mean([ic_list, ic_list2], axis=0), "-", color="red", linewidth=0.5)

    error = np.abs(np.subtract(ic_list, ic_list2)) / 2

    # ax = axs["D"]

    # ax.errorbar(write_current_list2, np.mean([ic_list, ic_list2], axis=0), yerr=error, fmt="o", color="black", markersize=3)
    ax = axs["D"]
    ax.plot(
        write_current_list2,
        np.mean([ic_list, ic_list2], axis=0),
        "o",
        color="black",
        markersize=3,
    )
    ax.set_ylim(axs["C"].get_ylim())
    ax.axhline(read_current, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 300)
    # ax.set_ylim(0, 1)
    # ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_ylabel("Optimal $I_{\mathrm{read}}$ [$\mu$A]")
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    
    fig.subplots_adjust(wspace=0.2, hspace=0.4)
    fig.patch.set_visible(False)

    plt.savefig("read_current_sweep_operating.pdf", bbox_inches="tight")
