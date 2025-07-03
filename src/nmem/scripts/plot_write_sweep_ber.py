#!/usr/bin/env python3
"""
Plot write current sweep bit error rate analysis.

This script analyzes bit error rate as a function of write current for
different operating conditions. Shows the relationship between write current
and memory cell fidelity, helping to determine optimal write parameters.
"""
import matplotlib.pyplot as plt

from nmem.analysis.data_import import load_and_process_write_sweep_data
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import plot_write_sweep_ber

# Apply global plot styling
apply_global_style()


def main(
    write_sweep_path="../data/ber_sweep_write_current/enable_write",
    save_dir=None,
):
    """
    Main function to generate write current sweep BER plots.
    """
    dict_list_ws = load_and_process_write_sweep_data(write_sweep_path)

    figsize = get_consistent_figure_size("single")
    fig, ax = plt.subplots(figsize=figsize)

    # Plot write sweep BER
    plot_write_sweep_ber(ax, dict_list_ws)

    if save_dir:
        fig.savefig(
            f"{save_dir}/write_sweep_ber.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
