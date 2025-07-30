#!/usr/bin/env python3
"""
Generate array fidelity bar plots showing bit error rates across memory array positions.

Creates both 3D and clean 2D bar visualizations of BER data to analyze
memory array performance and identify spatial patterns in error rates.
"""

import logging

from nmem.analysis.bar_plots import (
    plot_ber_3d_bar,
    plot_fidelity_clean_bar,
)
from nmem.analysis.core_analysis import process_ber_data
from nmem.analysis.styles import apply_global_style
import matplotlib.pyplot as plt
# Set plot styles
apply_global_style()

# Set up logger for better traceability
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def generate_plots(ber_array, save_dir=None):
    """
    Generate the plots and save them to the specified directory.
    """
    fig, ax = plot_ber_3d_bar(ber_array)
    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_3d_bar_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()

    fig, ax = plt.subplots()
    plot_fidelity_clean_bar(ber_array, ax=ax)
    if save_dir:
        plt.savefig(
            f"{save_dir}/fidelity_clean_bar_plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()

def main(save_dir=None):
    """
    Main function to process data and generate plots.
    """
    ber_array = process_ber_data(logger=logger)
    generate_plots(ber_array, save_dir)


if __name__ == "__main__":
    # Call the main function
    main()
