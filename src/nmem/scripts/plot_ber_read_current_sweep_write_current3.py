import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style, apply_legend_style
from nmem.analysis.sweep_plots import (
    plot_read_sweep_array,
)

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    dict_list = import_directory("../data/ber_sweep_read_current/write_current/data2")
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, dict_list[0:4], "bit_error_rate", "write_current")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("Read Current [$\\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    apply_legend_style(ax, "outside_right", title="Write Current [$\\mu$A]")

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_write_current3_part1.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()

    fig, ax = plt.subplots()
    dict_list2 = dict_list[4:]
    plot_read_sweep_array(ax, dict_list2, "bit_error_rate", "write_current")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("Read Current [$\\mu$A]")
    ax.set_ylabel("Bit Error Rate")
    apply_legend_style(ax, "outside_right", title="Write Current [$\\mu$A]")

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_write_current3_part2.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
