
import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import plot_read_sweep_array

plt.rcParams["figure.figsize"] = [5, 3.5]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    dict_list = import_directory("data")
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "write_width")
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Width",
    )
    plt.show()
    