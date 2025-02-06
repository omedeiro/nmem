import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_channel_temperature,
    plot_write_sweep,
)

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


if __name__ == "__main__":
    dict_list = import_directory("data")[1:]
    fig, axs = plt.subplots(1, 2, figsize=(6, 4), width_ratios=[1, 0.25])
    # dict_list = dict_list[::3]
    ax = axs[0]
    ax, ax2 = plot_write_sweep(ax, dict_list)
    # ax.legend(
    #     frameon=False,
    #     loc="upper left",
    #     bbox_to_anchor=(1.1, 1),
    #     title="Enable Write Temperature [K]",
    # )

    ax.set_xlabel("$I_{\mathrm{write}}$ ($\mu$A)")
    ax.set_ylabel("BER")
    # plt.savefig("write_current_sweep_enable_write.pdf", bbox_inches="tight")
    # plt.show()

    # fig, ax = plt.subplots()
    ax = axs[1]
    for data_dict in dict_list:
        plot_channel_temperature(ax, data_dict, marker="o", color="b")
    
    fig.subplots_adjust(wspace=0.3)
    plt.show()
