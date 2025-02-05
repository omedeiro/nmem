import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_channel_temperature,
    plot_write_sweep,
)

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


if __name__ == "__main__":
    dict_list = import_directory("data")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax, ax2 = plot_write_sweep(ax, dict_list[::-1])
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.1, 1),
        title="Enable Write Current",
    )
    plt.savefig("write_current_sweep_enable_write.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots()
    for data_dict in dict_list:
        plot_channel_temperature(ax, data_dict, marker="o", color="b")
    plt.show()
