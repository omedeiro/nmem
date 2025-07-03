"""
Write Current Enable Margin Markers Analysis

This script generates plots showing temperature markers vs write current,
displaying the operating margins and bounds for write current settings.
These markers help identify safe operating regions and enable current margins
for reliable memory write operations.
"""

import matplotlib.pyplot as plt

from nmem.analysis.data_import import (
    import_write_sweep_formatted,
    import_write_sweep_formatted_markers,
)
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import plot_write_sweep_formatted_markers

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    """
    Generate write current enable margin markers plot.

    Args:
        save_dir (str): Directory to save plots (if None, displays plots)
    """
    figsize = get_consistent_figure_size("single")
    fig, ax = plt.subplots(figsize=figsize)

    dict_list = import_write_sweep_formatted()
    data_dict = import_write_sweep_formatted_markers(dict_list)
    plot_write_sweep_formatted_markers(ax, data_dict)


    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_write_current_enable_margin_markers.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
