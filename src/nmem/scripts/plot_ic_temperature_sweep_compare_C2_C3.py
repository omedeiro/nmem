"""
This script compares C2 and C3 cells based on enable current relation data.
It generates two plots:
 1. A single axis comparison of C2 and C3 cells.
 2. Subplots for C3 cell data.
Despite being in the same column, the C2 and C3 cells exhibit different behaviors,
C3 has a step in the enable current relation, while C2 has a typical response.
"""

import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    get_fitting_points,
)
from nmem.analysis.data_import import import_directory
from nmem.analysis.htron_plotting import (
    plot_c2c3_comparison,
    plot_c3_subplots,
)
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.utils import build_array

# Apply global plot styling
apply_global_style()


def main(
    data_dir="../data/enable_current_relation/compare_C2_C3",
    save_dir=None,
):
    """
    Main function to compare C2 and C3 cells and plot results.
    """
    data_list = import_directory(data_dir)
    data_dict = data_list[0]
    data_dict2 = data_list[1]
    split_idx = 10
    # First plot: single axis comparison
    fig1, ax1 = plt.subplots()
    x, y, ztotal = build_array(data_dict, "total_switches_norm")
    x2, y2, ztotal2 = build_array(data_dict2, "total_switches_norm")

    c2 = get_fitting_points(x, y, ztotal)
    c3 = get_fitting_points(x2, y2, ztotal2)
    plot_c2c3_comparison(ax1, c2, c3, split_idx=6)
    # Second plot: subplots for C2 and C3
    figsize2 = get_consistent_figure_size("wide")
    fig2, axs2 = plt.subplots(1, 2, figsize=figsize2)
    plot_c3_subplots(axs2, c3, split_idx)
    if save_dir:
        fig1.savefig(
            f"{save_dir}/c2_c3_comparison.png",
            bbox_inches="tight",
            dpi=300,
        )
        fig2.savefig(
            f"{save_dir}/c3_subplots.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
