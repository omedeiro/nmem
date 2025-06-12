import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_read_sweep_array,
)


def main(
    data_dir="data", save_fig=False, output_path="read_current_sweep_read_width.pdf"
):
    dict_list = import_directory(data_dir)
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "read_width")
    ax.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(1, 1), title="Read Width"
    )
    ax.set_yscale("log")
    ax.set_ylabel("Bit Error Rate")
    ax.set_xlabel("Read Current ($\\mu$A)")
    ax.set_ylim([1e-4, 1])
    if save_fig:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
