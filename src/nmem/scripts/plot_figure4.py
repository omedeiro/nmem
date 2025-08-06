#!/usr/bin/env python3
"""
Figure 4: Comprehensive BER Analysis Summary

This script creates a 2x3 figure arran    # 6. Bottom center: 3D BER Bar Plot
    print("  - 3D BER Bar Plot")
    ber_array = process_ber_data(logger=logger)
    plot_ber_3d_bar(ber_array, ax=ax_bottom_center)
    ax_bottom_center.set_title("(f) 3D BER Array", fontweight="bold")

    # 7. Bottom right: Array Fidelity Bar showing key bit error rate (BER)
analysis results including enable/write current sweeps, memory retention, and
array fidelity measurements. The figure provides a comprehensive overview of
memory device performance characteristics across multiple measurement types.
"""

import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nmem.analysis.bar_plots import (
    plot_fidelity_clean_bar,
    plot_ber_3d_bar,
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
    # Set up the figure with mixed subplot arrangement
    figsize = get_consistent_figure_size("wide")
    fig = plt.figure(
        figsize=(120 / 25.4, 90 / 25.4)
    )  # Wide figure for 2x2 + right column

    # Create gridspec for mixed layout: 2x2 left section + 1x3 right column
    gs = gridspec.GridSpec(
        2,
        2,  # Use 5 columns: 2+2 for left 2x2, 1 for right column
        figure=fig,
        hspace=0.3,
        wspace=0.3,
    )

    # Create subplots
    # Left side: 2x2 grid spanning all 3 rows (first 2 columns)
    ax_top_left = fig.add_subplot(gs[0, 0])  # BER vs Enable Current (spans 2 rows)
    ax_top_center = fig.add_subplot(gs[0, 1])  # BER vs Write Current (spans 2 rows)
    ax_bottom_left = fig.add_subplot(gs[1, 0])  # State Current Markers
    ax_bottom_center = fig.add_subplot(gs[1, 1])  # Enable Margin Markers

    # Right column: 3 plots stacked vertically (column 3, with spacing)
    # ax_top_right = fig.add_subplot(gs[0:3, 2], projection="3d")  # Memory Retention
    # ax_mid_right = fig.add_subplot(gs[3:5, 2])  # 3D BER Plot
    # ax_bottom_right = fig.add_subplot(gs[5:6, 2])  # Array Fidelity

    # Load data for each plot
    print("Loading data for Figure 4 plots...")

    # 1. Top left: BER vs Enable Write Current Sweep (Figure 1)
    print("  - BER Enable Write Sweep (Figure 1)")
    dict_list_ews = import_directory("../data/ber_sweep_enable_write_current/data1")
    plot_enable_write_sweep_multiple(dict_list_ews, ax=ax_top_left, add_legend=False)
    ax_top_left.set_xlim([250, 335])
    ax_top_left.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax_top_left.xaxis.set_minor_locator(plt.MultipleLocator(5))

    # Add professional Nature-style cell label to top left plot
    ax_top_left.text(
        0.05,
        0.95,
        "Cell C1",
        transform=ax_top_left.transAxes,
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )

    # 2. Top center: BER vs Write Current Sweep
    print("  - BER Write Current Sweep")
    dict_list_ws = import_write_sweep_formatted()
    plot_write_sweep_formatted(ax_top_center, dict_list_ws, add_legend=False)
    ax_top_center.text(
        0.05,
        0.95,
        "Cell C3",
        transform=ax_top_center.transAxes,
        fontsize=7,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )
    # 3. Bottom left: BER Enable Write Sweep State Current Markers (Figure 2)
    print("  - BER Enable Write Sweep Markers (Figure 2)")
    plot_state_current_markers(dict_list_ews, ax=ax_bottom_left, add_legend=True)
    ax_bottom_left.set_xlim([250, 335])
    ax_bottom_left.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax_bottom_left.xaxis.set_minor_locator(plt.MultipleLocator(5))
    # 4. Bottom center: Write Current Enable Margin Markers
    print("  - Write Current Enable Margin Markers")
    data_dict = import_write_sweep_formatted_markers(dict_list_ws)
    plot_write_sweep_formatted_markers(ax_bottom_center, data_dict, add_legend=True)


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
