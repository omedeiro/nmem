
import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import (
    plot_critical_currents_from_dc_sweep,
)
from nmem.analysis.trace_plots import (
    plot_combined_dc_figure,
    plot_current_voltage_from_dc_sweep,
)
from nmem.analysis.styles import set_plot_style

set_plot_style()


def run_all_dc_sweep_plots(
    data_dir="data", save_combined=False, combined_path="iv_curve_combined.pdf"
):
    """
    Run all DC sweep plots and optionally save the combined figure.
    Returns all figures and axes for further use or testing.
    """
    data_list = import_directory(data_dir)

    fig1, ax1 = plt.subplots()
    plot_critical_currents_from_dc_sweep(ax1, data_list)
    fig2, ax2 = plt.subplots()
    plot_current_voltage_from_dc_sweep(ax2, data_list)
    fig3, axs = plt.subplots(1, 3, figsize=(7, 4))
    plot_combined_dc_figure(axs, data_list)
    if save_combined:
        plt.subplots_adjust(wspace=0.4)
        fig3.savefig(combined_path, bbox_inches="tight")
    return (fig1, ax1), (fig2, ax2), (fig3, axs)


if __name__ == "__main__":
    run_all_dc_sweep_plots()
    plt.show()
