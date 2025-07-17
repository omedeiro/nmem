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

    fig = plt.figure(
        figsize=(60 / 25.4, 90 / 25.4)
    )  # Wide figure for 2x2 + right column

    # Create gridspec for mixed layout: 2x2 left section + 1x3 right column
    gs = gridspec.GridSpec(
        3,
        1,  # Use 5 columns: 2+2 for left 2x2, 1 for right column
        figure=fig,
        hspace=0.6,
        wspace=0.3,
    )
    ax_top_right = fig.add_subplot(gs[0, 0], projection="3d")  # Memory Retention
    ax_mid_right = fig.add_subplot(gs[1, 0])  # 3D BER Plot
    ax_bottom_right = fig.add_subplot(gs[2, 0])  # Array Fidelity

    # 6. Middle right: 3D BER Bar Plot
    print("  - 3D BER Bar Plot")
    ber_array = process_ber_data(logger=logger)
    plot_ber_3d_bar(ber_array, ax=ax_top_right)

    # 7. Bottom right: Array Fidelity Bar
    print("  - Array Fidelity Bar Plot")
    plot_fidelity_clean_bar(ber_array, ax=ax_mid_right)

    # 5. Top right: Memory Retention
    print("  - Memory Retention")
    delay_list, bit_error_rate_list, _ = import_delay_data()
    plot_retention(delay_list, bit_error_rate_list, ax=ax_bottom_right)


    # pos = ax_top_right.get_position()
    # ax_top_right.set_position([pos.x0 - 0.15, pos.y0, pos.width, pos.height])


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
