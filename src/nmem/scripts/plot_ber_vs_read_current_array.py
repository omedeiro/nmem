#!/usr/bin/env python3
"""
Plot BER vs read current for different write currents.

This script analyzes bit error rate as a function of read current for various
write current setpoints. Shows the operating regions and switching behavior
of the memory cell under different read conditions.
"""
import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size, apply_legend_style
from nmem.analysis.sweep_plots import plot_read_sweep_array

# Apply global plot styling
apply_global_style()


def main(
    data_dir="../data/ber_sweep_read_current/write_current/write_current_sweep_C3",
    save_dir=None,
):
    """
    Main function to generate BER vs read current plots.
    """
    dict_list = import_directory(data_dir)

    figsize = get_consistent_figure_size("single")
    fig, ax = plt.subplots(figsize=figsize)

    # Plot BER vs read current for different write currents
    plot_read_sweep_array(
        ax,
        dict_list,
        "bit_error_rate",
        "write_current",
    )
    ax.set_xlim(650, 850)
    ax.set_xlabel("$I_{\\mathrm{read}}$ [$\\mu$A]", labelpad=-3)
    ax.set_ylabel("BER")
    apply_legend_style(ax, "outside_right", title="Write Current [$\\mu$A]")
    if save_dir:
        fig.savefig(
            f"{save_dir}/ber_vs_read_current_array.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
