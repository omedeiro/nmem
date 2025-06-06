import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_read_sweep_array,
)

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    data_list = import_directory("data")
    data_list = data_list[::2]
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_list, "bit_error_rate", "write_current")
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\mu$A]",
    )
    save = False
    if save:
        plt.savefig(
            "read_current_sweep_write_current.png", dpi=300, bbox_inches="tight"
        )
    plt.show()

    data_list2 = import_directory("data2")
    data_list2 = data_list2[::2]
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_list2, "bit_error_rate", "write_current")
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\mu$A]",
    )
    plt.show()
