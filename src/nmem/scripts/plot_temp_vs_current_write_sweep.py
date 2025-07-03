#!/usr/bin/env python3
"""
Plot temperature vs current analysis from write sweep data.

This script analyzes the relationship between operating temperature and
current for different write sweep conditions. Provides insights into
thermal effects on memory cell operation and critical current behavior.
"""
import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import extract_temp_current_data
from nmem.analysis.data_import import load_and_process_write_sweep_data
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import plot_temp_vs_current

# Apply global plot styling
apply_global_style()


def main(
    write_sweep_path="../data/ber_sweep_write_current/enable_write",
    save_dir=None,
):
    """
    Main function to generate temperature vs current plots.
    """
    dict_list_ws = load_and_process_write_sweep_data(write_sweep_path)

    figsize = get_consistent_figure_size("single")
    fig, ax = plt.subplots(figsize=figsize)

    # Extract temperature and current data
    data, data2 = extract_temp_current_data(dict_list_ws)

    # Plot temperature vs current
    plot_temp_vs_current(ax, data, data2)

    if save_dir:
        fig.savefig(
            f"{save_dir}/temp_vs_current_write_sweep.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
