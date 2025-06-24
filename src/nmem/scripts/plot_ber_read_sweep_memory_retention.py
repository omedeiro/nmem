import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import (
    plot_read_delay,
)


def main(save_dir=None):
    dict_list = import_directory("../data/ber_sweep_read_current/memory_retention")

    fig, ax = plt.subplots()
    plot_read_delay(ax, dict_list)
    ax.set_ylim(1e-4, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Read Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Read Delay",
    )

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_sweep_memory_retention.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
