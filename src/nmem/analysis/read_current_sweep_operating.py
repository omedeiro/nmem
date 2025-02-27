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
    RBCOLORS,
    WIDTH,
    SUBSTRATE_TEMP,
    CRITICAL_TEMP,
    calculate_retrapping_current_temp,
)

CRITICAL_CURRENT_ZERO = 1250
IRM = 727.5
IRHL_TR = 105
MAX_IWRITE = 300
MAX_IP = 110

def calculate_inductance_ratio(state0, state1, ic0):
    alpha = (ic0 - state1) / (state0 - state1)
    return alpha


if __name__ == "__main__":
    fig, axs = plt.subplot_mosaic("AA;CD", figsize=(8.3, 4))

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
    # ax.axvline(IRM, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]", labelpad=-3)
    ax.set_ylabel("BER")
    ax.set_xlim(500, 850)
    ax2 = ax.twinx()
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Switching Probability")
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))

    ax = axs["C"]
    for data_dict in dict_list:
        state_current_markers = get_state_current_markers(data_dict, "read_current")
        write_current = get_write_current(data_dict)
        for i, state_current in enumerate(state_current_markers[0, :]):
            if state_current > 0:
                ax.plot(
                    write_current,
                    state_current,
                    "o",
                    label=f"{write_current} $\mu$A",
                    markerfacecolor=RBCOLORS[i],
                    markeredgecolor="none",
                    markersize=4,
                )
    ax.set_xlim(0, write_current)
    ax.set_ylabel("$I_{\mathrm{state}}$ [$\mu$A]")
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")

    ic_list = [IRM]
    write_current_list = [0]
    ic_list2 = [IRM]
    write_current_list2 = [0]
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
    ax.set_xlim(0, MAX_IWRITE)
    ax.set_ylabel("$I_{\mathrm{read}}$ [$\mu$A]")
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.axhline(IRM, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(IRHL_TR, color="black", linestyle="--", linewidth=0.5)
    # ax.axvline(IRHL_TR/2, color="black", linestyle="--", linewidth=0.5)

    persistent_current = []
    upper = []
    lower = []
    for i, write_current in enumerate(write_current_list):
        if write_current > IRHL_TR / 2:
            ip = np.abs(write_current - IRHL_TR)
        else:
            ip = write_current
        if ip > IRHL_TR:
            ip = IRHL_TR
        # ax.vlines(
        #     write_current,
        #     ymin=IRM - ip,
        #     ymax=IRM + ip,
        #     color="black",
        #     linestyle="--",
        #     linewidth=0.5,
        # )
    # write_current_array = np.array(write_current_list)
    write_current_array = np.linspace(
        write_current_list[0], write_current_list[-1], 1000
    )
    persistent_current = np.where(
        write_current_array > IRHL_TR / 2,
        np.abs(write_current_array - IRHL_TR),
        write_current_array,
    )
    persistent_current = np.where(
        persistent_current > IRHL_TR, IRHL_TR, persistent_current
    )
    upper = IRM + persistent_current / 2
    lower = IRM - persistent_current / 2
    ax.fill_between(write_current_array, lower, upper, color="black", alpha=0.1)
    ic = np.array(ic_list)
    ic2 = np.array(ic_list2)

    read_temperature = calculate_channel_temperature(
        CRITICAL_TEMP,
        SUBSTRATE_TEMP,
        data_dict["enable_read_current"] * 1e6,
        CELLS[data_dict["cell"][0]]["x_intercept"],
    ).flatten()
    write_temperature = calculate_channel_temperature(
        CRITICAL_TEMP,
        SUBSTRATE_TEMP,
        data_dict["enable_write_current"] * 1e6,
        CELLS[data_dict["cell"][0]]["x_intercept"],
    ).flatten()

    delta_read_current = np.subtract(ic2, ic)

    critical_current_channel = calculate_critical_current_temp(
        read_temperature, CRITICAL_TEMP, CRITICAL_CURRENT_ZERO
    )
    critical_current_left = critical_current_channel * WIDTH
    critical_current_right = critical_current_channel * (1 - WIDTH)

    retrap = (ic - critical_current_left) / critical_current_right
    retrap2 = (ic2 - critical_current_right) / critical_current_left

    left_retrapping_current = critical_current_left * retrap2
    right_retrapping_current = critical_current_right * retrap2

    ax = axs["D"]
    ax.plot(write_current_list, np.abs(delta_read_current), "-o", color="black", markersize=3.5)
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$|\Delta I_{\mathrm{read}}|$ [$\mu$A]")
    ax.set_xlim(0, MAX_IWRITE)
    ax.set_ylim(0, MAX_IP)
    ax.patch.set_alpha(0)
    ax.set_zorder(1)

    ax2 = ax.twinx()
    ax2.plot(write_current_array, persistent_current, "-", color="grey", zorder=-1)
    ax2.set_ylabel("$I_{\mathrm{persistent}}$ [$\mu$A]")
    ax2.set_ylim(0, MAX_IP)
    ax2.set_zorder(0)
    ax2.fill_between(write_current_array, np.zeros_like(write_current_array), persistent_current, color="black", alpha=0.1)
    fig.subplots_adjust(wspace=0.33, hspace=0.4)
    plt.savefig("read_current_sweep_operating.pdf", bbox_inches="tight")
    plt.show()
