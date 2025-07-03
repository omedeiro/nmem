#!/usr/bin/env python3
"""
Plot Bit Error Rate (BER) analysis for geometric loop size sweep.

This script generates two plots:
1. BER vs channel voltage for different loop sizes
2. Minimum BER vs loop size

The geometric loop size parameter (w_5) affects the memory cell's switching
characteristics and BER performance. These measurements were performed in the probe station
at a base temperature of 3.5K.
The estimated BER was calculated by fitting two gaussian functions to the data and taking the intersection of the fitted curves.

The data here is insufficient to determine the optimal loop size, but it provides a good starting point for further analysis.
"""

import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import analyze_geom_loop_size
from nmem.analysis.data_import import import_geom_loop_size_data
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import plot_loop_size_sweep, plot_loop_size_sweep_ber

# Apply global plot styling
apply_global_style()


def main(data_dir="../data/loop_size_sweep", save_dir=None):
    data, loop_sizes = import_geom_loop_size_data(data_dir)
    vch_list, ber_est_list, err_list, best_ber = analyze_geom_loop_size(
        data, loop_sizes
    )

    # Create subplot layout with both plots
    figsize = get_consistent_figure_size("wide")
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # First plot: Vch vs. ber_est
    plot_loop_size_sweep(data, vch_list, ber_est_list, loop_sizes, ax=axs[0])

    # Second plot: best BER vs loop size
    plot_loop_size_sweep_ber(loop_sizes, best_ber, ax=axs[1])

    # Adjust layout
    plt.tight_layout()

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_geom_loop_size.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
