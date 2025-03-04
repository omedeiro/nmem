import matplotlib.pyplot as plt

from nmem.analysis.analysis import import_directory, plot_read_sweep_array

def plot_read_sweep_import(dict_list):
    fig, ax = plt.subplots()
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
    plt.show()
    return fig, ax


if __name__ == "__main__":
    dict_list = import_directory("data2")
    plot_read_sweep_import(dict_list)

    dict_list = dict_list[4:]
    dict_list = dict_list[::-1]
    plot_read_sweep_import(dict_list)

    plot_read_sweep_import(import_directory("data3"))

    plot_read_sweep_import(import_directory("data4"))