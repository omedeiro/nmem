#!/usr/bin/env python3
"""
Figure 4: Comprehensive BER Analysis Summary

This script creates a 2x3 figure arrangement showing key bit error rate (BER)
analysis results including enable/write current sweeps, memory retention, and
array fidelity measurements. The figure provides a comprehensive overview of
memory device performance characteristics across multiple measurement types.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nmem.analysis.bar_plots import (
    plot_fidelity_clean_bar,
)
from nmem.analysis.core_analysis import process_ber_data
from nmem.analysis.data_import import (
    import_delay_data,
    import_directory,
    import_write_sweep_formatted,
    import_write_sweep_formatted_markers,
)
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import (
    plot_enable_write_sweep_multiple,
    plot_retention,
    plot_state_current_markers,
    plot_write_sweep_formatted,
    plot_write_sweep_formatted_markers,
)

# Apply global plot styling
apply_global_style()

# Set up logger for better traceability
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(save_dir=None):
    """
    Generate Figure 4 with comprehensive BER analysis.

    Args:
        save_dir (str): Directory to save plots (if None, displays plots)
    """
    # Set up the figure with 3x2 subplot arrangement
    figsize = get_consistent_figure_size("wide")
    fig = plt.figure(figsize=(15, 12))  # Adjusted for 3-column layout

    # Create gridspec for 3x2 arrangement with custom spacing
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1, 1, 1],
        hspace=0.3,
        wspace=0.3,
    )

    # Create subplots
    ax_top_left = fig.add_subplot(gs[0, 0])  # plot_ber_enable_write_sweep figure 1
    ax_top_right = fig.add_subplot(gs[0, 1])  # plot_ber_write_current_sweep
    ax_mid_left = fig.add_subplot(gs[1, 0])  # plot_ber_enable_write_sweep figure 2
    ax_mid_right = fig.add_subplot(
        gs[1, 1]
    )  # plot_ber_write_current_enable_margin_markers
    ax_bottom_left = fig.add_subplot(gs[2, 0])  # plot_ber_memory_retention
    ax_bottom_right = fig.add_subplot(
        gs[2, 1]
    ) 

    # Load data for each plot
    print("Loading data for Figure 4 plots...")

    # 1. Top left: BER vs Enable Write Current Sweep (Figure 1)
    print("  - BER Enable Write Sweep (Figure 1)")
    dict_list_ews = import_directory("../data/ber_sweep_enable_write_current/data1")
    plot_enable_write_sweep_multiple(dict_list_ews, ax=ax_top_left)
    ax_top_left.set_title("(a) BER vs Enable Current", fontweight="bold")

    # 2. Top right: BER vs Write Current Sweep
    print("  - BER Write Current Sweep")
    dict_list_ws = import_write_sweep_formatted()
    plot_write_sweep_formatted(ax_top_right, dict_list_ws)
    ax_top_right.set_title("(b) BER vs Write Current", fontweight="bold")

    # 3. Middle left: BER Enable Write Sweep State Current Markers (Figure 2)
    print("  - BER Enable Write Sweep Markers (Figure 2)")
    plot_state_current_markers(dict_list_ews, ax=ax_mid_left)
    ax_mid_left.set_title("(c) State Current Markers", fontweight="bold")

    # 4. Middle right: Write Current Enable Margin Markers
    print("  - Write Current Enable Margin Markers")
    data_dict = import_write_sweep_formatted_markers(dict_list_ws)
    plot_write_sweep_formatted_markers(ax_mid_right, data_dict)
    ax_mid_right.set_title("(d) Enable Margin Markers", fontweight="bold")

    # 5. Bottom left: Memory Retention
    print("  - Memory Retention")
    delay_list, bit_error_rate_list, _ = import_delay_data()
    plot_retention(delay_list, bit_error_rate_list, ax=ax_bottom_left)
    ax_bottom_left.set_title("(e) Memory Retention", fontweight="bold")

    # 6. Right side: Array Fidelity Bar (spanning two rows)
    print("  - Array Fidelity Bar Plot")
    ber_array = process_ber_data(logger=logger)
    plot_fidelity_clean_bar(ber_array, ax=ax_bottom_right)
    ax_bottom_right.set_title("(f) Array Fidelity", fontweight="bold")

    # Adjust layout
    plt.tight_layout()

    # Save or show the figure
    if save_dir:
        plt.savefig(
            f"{save_dir}/figure4_comprehensive_ber_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Figure 4 saved to {save_dir}/figure4_comprehensive_ber_analysis.png")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
