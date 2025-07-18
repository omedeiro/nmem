#!/usr/bin/env python3
"""
Figure 4 Right Panel: BER Analysis Summary

This script creates a 3x1 vertical figure arrangement showing key bit error rate (BER)
analysis results including 3D BER visualization, array fidelity measurements, and
memory retention characteristics. The figure provides a focused overview of
memory device performance in a compact vertical layout.
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


def main(save_dir="../plots"):

    fig = plt.figure(figsize=(60 / 25.4, 90 / 25.4))  # Vertical figure for 3x1 layout

    # Create gridspec for 3x1 vertical layout with custom height ratios
    gs = gridspec.GridSpec(
        3,
        1,  # 3 rows, 1 column
        figure=fig,
        hspace=0.4,
        wspace=0.3,
        height_ratios=[1, 1, 1],  # Give more space to the 3D plot
    )
    ax_top_right = fig.add_subplot(gs[0, 0], projection="3d")  # 3D BER Bar Plot
    ax_mid_right = fig.add_subplot(gs[1, 0])  # Array Fidelity Bar
    ax_bottom_right = fig.add_subplot(gs[2, 0])  # Memory Retention

    # 1. Top: 3D BER Bar Plot
    print("  - 3D BER Bar Plot")
    ber_array = process_ber_data(logger=logger)
    plot_ber_3d_bar(ber_array, ax=ax_top_right)

    # 2. Middle: Array Fidelity Bar
    print("  - Array Fidelity Bar Plot")
    plot_fidelity_clean_bar(ber_array, ax=ax_mid_right)

    # 3. Bottom: Memory Retention
    print("  - Memory Retention")
    delay_list, bit_error_rate_list, _ = import_delay_data()
    plot_retention(delay_list, bit_error_rate_list, ax=ax_bottom_right)

    # Adjust 3D plot position and size to fill more space
    pos = ax_top_right.get_position()
    ax_top_right.set_position(
        [pos.x0 - 0.25, pos.y0 - 0.05, pos.width + 0.4, pos.height + 0.2]
    )

    ax_bottom_right.text(
        0.95,
        0.05,
        "Cell C3",
        transform=ax_bottom_right.transAxes,
        fontsize=7,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )
    # Save or show the figure
    if save_dir:
        plt.savefig(
            f"{save_dir}/figure4_right_panel_ber_analysis.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
