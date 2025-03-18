import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    CMAP,
    get_channel_temperature,
    get_enable_read_current,
    get_enable_write_current,
    import_directory,
    plot_fill_between,
    plot_fill_between_array,
    plot_read_sweep,
    plot_read_sweep_array,
)

if __name__ == "__main__":
    data = import_directory("data")

    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    enable_read_310_C4_list = import_directory("data_310uA_C4")

    data_inverse = import_directory("data_inverse")

    dict_list = [enable_read_290_list, enable_read_300_list, enable_read_310_list]
    dict_list = dict_list[2]

    read_temperatures = []
    enable_read_currents = []
    for data_dict in dict_list:
        read_temperature = get_channel_temperature(data_dict, "read")
        enable_read_current = get_enable_read_current(data_dict)
        read_temperatures.append(read_temperature)
        enable_read_currents.append(enable_read_current)

    fig, axs = plt.subplots(
        2, 2, figsize=(8.3, 4), constrained_layout=True, width_ratios=[1, 0.25]
    )

    ax = axs[1, 0]
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "enable_read_current")
    plot_fill_between_array(ax, dict_list)
    ax.axvline(910, color="black", linestyle="--")
    ax.set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(400, 1000)

    ax = axs[1, 1]
    ax.plot(
        enable_read_currents,
        read_temperatures,
        marker="o",
        color="black",
    )
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    ax.set_ylabel("$T_{\mathrm{read}}$ [K]")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    data_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\read_current_sweep_enable_write\data"
    )
    data_list2 = [data_list[0], data_list[3], data_list[-6], data_list[-1]]
    colors = CMAP(np.linspace(0, 1, len(data_list2)))

    ax = axs[0, 0]
    for j, data_dict in enumerate(data_list2):
        plot_read_sweep(
            ax, data_dict, "bit_error_rate", "enable_write_current", color=colors[j]
        )
        plot_fill_between(ax, data_dict, fill_color=colors[j])
        enable_write_current = get_enable_write_current(data_dict)

    ax.set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlim(400, 1000)
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))

    ax = axs[0, 1]
    enable_write_currents = []
    write_temperatures = []
    for i, data_dict in enumerate(data_list):
        #     state_currents = get_state_currents_measured(data_dict, "enable_write_current")
        #     for j, state in enumerate(state_currents[1]):
        #         if state > 0:
        #             ax.plot(
        #                 state_currents[0],
        #                 state,
        #                 marker="o",
        #                 color=colors[j],
        #             )
        enable_write_current = get_enable_write_current(data_dict)
        write_temperature = get_channel_temperature(data_dict, "write")
        enable_write_currents.append(enable_write_current)
        write_temperatures.append(write_temperature)
    # ax.set_xlabel("Write Temperature [K]")
    # ax.set_ylabel("Read Current [$\mu$A]")

    ax.plot(
        enable_write_currents,
        write_temperatures,
        marker="o",
        color="black",
    )

    for i, idx in enumerate([0, 3, -6, -1]):
        ax.plot(
            enable_write_currents[idx],
            write_temperatures[idx],
            marker="o",
            markersize=6,
            markeredgecolor="none",
            markerfacecolor=colors[i],
        )
    ax.set_ylabel("$T_{\mathrm{write}}$ [K]")
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    plt.savefig("read_current_sweep_enable_read3.pdf", bbox_inches="tight")
