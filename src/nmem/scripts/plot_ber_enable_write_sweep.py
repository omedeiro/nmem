"""
Bit Error Rate vs Enable/Write Current Sweep Analysis

This script analyzes and visualizes the relationship between bit error rate (BER)
and enable/write current combinations. It generates plots showing how BER varies
with different current settings, helping to identify optimal operating parameters
for memory write operations.
"""

import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_enable_write_sweep_multiple

# Apply global plot styling
apply_global_style()


def main(data_dir="../data/ber_sweep_enable_write_current/data1", save_dir=None):
    """
    Generate BER vs enable/write current sweep plot.

    Args:
        data_dir (str): Directory containing the measurement data
        save_dir (str): Directory to save plots (if None, displays plots)
    """
    dict_list = import_directory(data_dir)

    # Plot enable write sweep
    fig, ax = plt.subplots()
    fig, ax = plot_enable_write_sweep_multiple(dict_list, ax=ax)
    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_enable_write_sweep.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
