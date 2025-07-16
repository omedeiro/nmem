import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import plot_enable_write_sweep_multiple

# Apply global plot styling
apply_global_style()

IDX = 7
def main(save_dir=None):
    figsize = get_consistent_figure_size("single")
    dict_list = import_directory("../data/ber_sweep_enable_write_current/data1")
    sort_dict_list = sorted(
        dict_list, key=lambda x: x.get("write_current").flatten()[0]
    )

    # First figure - first half of data
    fig1, ax1 = plt.subplots(figsize=figsize)
    plot_enable_write_sweep_multiple(
        sort_dict_list[:IDX],
        ax = ax1,
    )
    ax1.set_title("BER vs Enable Current (Part 1)")

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_enable_write_sweep_supplemental_part1.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig1)
    else:
        plt.show()

    # Second figure - second half of data
    fig2, ax2 = plt.subplots(figsize=figsize)
    plot_enable_write_sweep_multiple(
        sort_dict_list[IDX:],
        ax=ax2,
    )
    ax2.set_title("BER vs Enable Current (Part 2)")

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_enable_write_sweep_supplemental_part2.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig2)
    else:
        plt.show()


if __name__ == "__main__":
    main()
