import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
)


def plot_read_sweep(ax, data_list):
    plot_read_sweep_array(ax, data_list, "bit_error_rate", "write_current")
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\mu$A]",
    )

    return ax


if __name__ == "__main__":
    # Import
    data_list = import_directory("data")
    data_list = data_list[::2]

    data_list2 = import_directory("data2")
    data_list2 = data_list2[::2]

    # Plot 1
    fig, ax = plt.subplots()
    plot_read_sweep(ax, data_list)
    save = False
    if save:
        fig.savefig(
            "read_current_sweep_write_current.png", dpi=300, bbox_inches="tight"
        )
    plt.show()

    # Plot 2
    fig, ax = plt.subplots()
    plot_read_sweep(ax, data_list2)
    plt.show()
