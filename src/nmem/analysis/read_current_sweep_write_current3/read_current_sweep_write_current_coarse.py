import matplotlib.pyplot as plt

from nmem.analysis.analysis import import_directory, plot_read_sweep_array

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


if __name__ == "__main__":
    fig, ax = plt.subplots()
    dict_list = import_directory("data4")
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "write_current")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [$\mu$A]",
    )
