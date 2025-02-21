import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
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
    calculate_critical_current_temp,
    calculate_channel_temperature,
    CELLS,
)

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


CRITICAL_CURRENT_ZERO = 1250
WIDTH = 0.3

CENTER_POINT=727 

def calculate_inductance_ratio(state0, state1, ic0):
    alpha = (ic0-state1)/(state0-state1)
    # alpha_test = 1 - ((critical_current_right - persistent_current_est) / ic)
    # alpha_test2 = (critical_current_left - persistent_current_est) / ic2
 
    return alpha


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
    # ax.axvline(read_current, color="black", linestyle="--", linewidth=1)
    
    ax.axvline(CENTER_POINT, color="black", linestyle="--", linewidth=1)
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
    ax.axhline(CENTER_POINT, color="black", linestyle="--", linewidth=1, label="Center Point")

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
        if not np.isnan(berargs[0]) and write_current < 100:
            ic_list.append(read_currents[berargs[0]])
            write_current_list.append(write_current)
        if not np.isnan(berargs[2]) and write_current > 100:
            ic_list.append(read_currents[berargs[3]])
            write_current_list.append(write_current)

        if not np.isnan(berargs[1]):
            ic_list2.append(read_currents[berargs[1]])
            write_current_list2.append(write_current)
        if not np.isnan(berargs[3]):
            ic_list2.append(read_currents[berargs[2]])
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
    # fig.patch.set_visible(False)
    # ax.plot(write_current_list2, np.mean([ic_list, ic_list2], axis=0), "-", color="red", linewidth=0.5)

    ic = np.array(ic_list)
    ic2 = np.array(ic_list2)
    write_current_array = np.array(write_current_list)
    read_temperature = calculate_channel_temperature(
        CRITICAL_TEMP,
        SUBSTRATE_TEMP,
        data_dict["enable_read_current"] * 1e6,
        CELLS[data_dict["cell"][0]]["x_intercept"],
    ).flatten()

    delta_read_current = np.subtract(ic2, ic)

    critical_current_channel = calculate_critical_current_temp(
        read_temperature, CRITICAL_TEMP, CRITICAL_CURRENT_ZERO
    )
    critical_current_left = critical_current_channel * WIDTH
    critical_current_right = critical_current_channel * (1 - WIDTH)

    alpha = calculate_inductance_ratio(ic, ic2, CENTER_POINT)

    retrap = (ic - critical_current_left) / critical_current_right
    retrap2 = (ic2 - critical_current_right) / critical_current_left

    left_retrapping_current = critical_current_left * retrap
    right_retrapping_current = critical_current_right * retrap2

    persistent_current_est = np.where(write_current_array*alpha > critical_current_left,-(write_current_array-left_retrapping_current[0]), write_current_array)
    # persistent_current_est = np.where(persistent_current_est<-left_retrapping_current[0], -left_retrapping_current, persistent_current_est)
    # persistent_current_est = np.where(persistent_current_est>left_retrapping_current[0], left_retrapping_current, persistent_current_est)


    i0 = critical_current_right + left_retrapping_current


    pd = pd.DataFrame(
        {
            "Write Current": write_current_list,
            "Read Current": ic_list,
            "Read Current 2": ic_list2,
            "Delta Read Current": delta_read_current,
            "Inductance Ratio": alpha,
            "Retrap": retrap,
            "Persistent Current": persistent_current_est,
            "Left Retrapping Current": left_retrapping_current,
            "Right Retrapping Current": right_retrapping_current,
            "I0": i0,
        }
    )
    print(pd)

    ax = axs["A"]
    ax.axvline(i0[0], color="black", linestyle="--", linewidth=1, label="i0")
    ax.axvline(critical_current_right, color="red", linestyle="--", linewidth=1, label="i0")
    ax.axvline(critical_current_channel, color="green", linestyle="--", linewidth=1, label="i0")
    ax = axs["C"]
    ax.axhline(i0[0], color="black", linestyle="--", linewidth=1, label="i0")
    # ax.axhline(critical_current_right, color="red", linestyle="--", linewidth=1, label="i0")
    # ax.axhline(critical_current_channel, color="green", linestyle="--", linewidth=1, label="i0")
    
    # Average inductance ratio
    avgalpha = pd["Inductance Ratio"].mean()
    print(f"Average inductance ratio: {avgalpha}")
    avgretrap = pd["Retrap"].mean()
    print(f"Average retrap: {avgretrap}")

    ax = axs["D"]
    ax.plot(
        np.append(0, write_current_list),
        np.append(0, persistent_current_est),
        "-o",
        color="black",
        markersize=3,
    )
    ax.set_xlim(0, 300)
    ax.set_ylabel("$I_{\mathrm{persistent}}$ [$\mu$A]")
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")


    fig.subplots_adjust(wspace=0.33, hspace=0.4)
    plt.show()