import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import plot_read_sweep_array

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


def main():
    fig, ax = plt.subplots()
    dict_list = import_directory("../data/ber_sweep_read_current/write_current/fine_sweep")
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "write_current")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\\mu$A]",
    )


if __name__ == "__main__":
    main()
