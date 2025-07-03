#!/usr/bin/env python3
"""
Plot write temperature vs current analysis.

This script analyzes the relationship between write current and temperature
effects on memory cell operation. Shows how write current affects the
effective temperature of the memory element and critical current behavior.
"""
import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import process_write_temp_arrays
from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import plot_write_temp_vs_current

# Apply global plot styling
apply_global_style()


def main(
    enable_write_sweep_path="../data/ber_sweep_enable_write_current/data1",
    save_dir=None,
):
    """
    Main function to generate write temperature vs current plots.
    """
    dict_list_ews = import_directory(enable_write_sweep_path)

    figsize = get_consistent_figure_size("single")
    fig, ax = plt.subplots(figsize=figsize)

    # Process write temperature arrays
    write_current_array, write_temp_array, critical_current_zero = (
        process_write_temp_arrays(dict_list_ews)
    )

    # Plot write temperature vs current
    plot_write_temp_vs_current(
        ax, write_current_array, write_temp_array, critical_current_zero
    )

    if save_dir:
        fig.savefig(
            f"{save_dir}/write_temp_vs_current.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
