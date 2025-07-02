"""
BER vs Write Current Sweep Analysis

This script generates plots showing the relationship between bit error rate (BER)
and write current values. It visualizes how BER varies across different write
current settings, helping to identify optimal write current ranges for reliable
memory operations.
"""

import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_write_sweep_formatted
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import plot_write_sweep_formatted

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    """
    Generate BER vs write current sweep plot.

    Args:
        save_dir (str): Directory to save plots (if None, displays plots)
    """
    figsize = get_consistent_figure_size("single")
    fig, ax = plt.subplots(figsize=figsize)

    dict_list = import_write_sweep_formatted()
    plot_write_sweep_formatted(ax, dict_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_write_current_sweep.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
