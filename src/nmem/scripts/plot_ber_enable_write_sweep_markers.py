"""
BER Enable/Write Current Sweep - State Current Markers Analysis

This script generates state current marker plots that show specific operating points
extracted from BER vs enable/write current sweep measurements. These markers help
identify optimal current combinations for reliable memory operations by highlighting
key state transitions and operating regions.
"""

import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_state_current_markers2

# Apply global plot styling
apply_global_style()


def main(data_dir="../data/ber_sweep_enable_write_current/data1", save_dir=None):
    """
    Generate state current markers plot from BER sweep data.

    Args:
        data_dir (str): Directory containing the measurement data
        save_dir (str): Directory to save plots (if None, displays plots)
    """
    dict_list = import_directory(data_dir)

    # Plot state current markers
    fig, ax = plot_state_current_markers2(dict_list)
    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_enable_write_sweep_state_current_markers.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
