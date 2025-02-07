import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep,
    plot_state_current_markers,
    get_state_currents_measured,
    get_write_current,
    get_enable_write_current,
    get_enable_write_current_array,
    get_write_temperatures_array,
    plot_fill_between,
    CMAP,
)

if __name__ == "__main__":
    data_list = import_directory("data")
    data_list2 = [data_list[0], data_list[3], data_list[-6]]
    colors = CMAP(np.linspace(0, 1, 4))

    fig, axs = plt.subplots(1,2, figsize=(8.37, 2), constrained_layout=True, width_ratios=[1, .25])


    ax = axs[0]
    for j, data_dict in enumerate(data_list2):
        plot_read_sweep(
            ax, data_dict, "bit_error_rate", "enable_write_current", color=colors[j]
        )
        # plot_state_current_markers(ax, data_dict, "enable_write_current")
        plot_fill_between(ax, data_dict, fill_color=colors[j])
        enable_write_current = get_enable_write_current(data_dict)
    # ax.legend(
    #     loc="upper right",
    #     bbox_to_anchor=(1, 1),
    # )
    ax.set_xlabel("$I_{\mathrm{read}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    # plt.show()
    ax.set_xlim(400, 1000)
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))

    ax = axs[1]
    # for i, data_dict in enumerate(data_list):
    #     state_currents = get_state_currents_measured(data_dict, "enable_write_current")
    #     for j, state in enumerate(state_currents[1]):
    #         if state > 0:
    #             ax.plot(
    #                 state_currents[0],
    #                 state,
    #                 marker="o",
    #                 color=colors[j],
    #             )
    write_current = get_write_current(data_dict)
    enable_write_currents = get_enable_write_current_array(data_list)
    # ax.set_xlabel("Write Temperature [K]")
    # ax.set_ylabel("Read Current [$\mu$A]")

    write_temperatures = get_write_temperatures_array(data_list)
    ax.plot(
        enable_write_currents,
        write_temperatures,
        marker="o",
        color="black",
    )

    for i, idx in enumerate([0, 3, -6]):
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
    ax.yaxis.set_major_locator(plt.MultipleLocator(.2))
    plt.savefig("read_current_sweep_enable_write2.pdf", bbox_inches="tight")
    plt.show()
