import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_read_sweep_array,
)

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    data_list = import_directory("data")

    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_list, "bit_error_rate", "enable_write_current")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    plt.show()
