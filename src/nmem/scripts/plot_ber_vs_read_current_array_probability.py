#!/usr/bin/env python3
"""
Plot switching probability vs read current analysis.

This script analyzes switching probability as a function of read current
for memory cells. Shows the probability of state switching at different
read current levels, providing insights into read stability.


See primary sweep: plot_ber_vs_read_current_array.py

"""
import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import (
    apply_global_style,
    apply_legend_style,
    get_consistent_figure_size,
)
from nmem.analysis.sweep_plots import plot_read_switch_probability_array

# Apply global plot styling
apply_global_style()


def main(
    data_dir="../data/ber_sweep_read_current/write_current/write_current_sweep_C3",
    save_dir=None,
):
    """
    Main function to generate switching probability vs read current plots.
    """
    dict_list = import_directory(data_dir)

    figsize = get_consistent_figure_size("single")
    fig, ax = plt.subplots(figsize=figsize)

    # Plot switching probability vs read current
    plot_read_switch_probability_array(ax, dict_list)
    ax.set_xlim(650, 850)
    ax.set_xlabel("$I_{\\mathrm{read}}$ [$\\mu$A]", labelpad=-3)
    ax.set_ylabel("Switching Probability")
    apply_legend_style(ax, "outside_right", title="Write Current [$\\mu$A]")
    if save_dir:
        fig.savefig(
            f"{save_dir}/ber_vs_read_current_array_probability.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
