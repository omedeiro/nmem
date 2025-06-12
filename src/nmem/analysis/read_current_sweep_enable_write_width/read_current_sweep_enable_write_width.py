import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import plot_read_sweep_array


def main(
    data_dir="data",
    save_fig=False,
    output_path="read_current_sweep_enable_write_width.pdf",
):
    dict_list = import_directory(data_dir)
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "enable_write_width")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Enable Write Width [pts]",
    )
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 1])
    ax.set_xlabel("Read Current ($\\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    if save_fig:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
