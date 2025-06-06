from typing import List

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    plot_critical_currents_from_dc_sweep,
    plot_current_voltage_from_dc_sweep,
)
from nmem.analysis.styles import set_plot_style
from nmem.analysis.constants import PROBE_STATION_TEMP

set_plot_style()


def plot_combined_figure(axs: List[Axes], dict_list: list) -> List[Axes]:
    """
    Plot combined IV and critical current figures on provided axes.
    """
    axs[0].set_axis_off()
    plot_current_voltage_from_dc_sweep(axs[1], dict_list)
    plot_critical_currents_from_dc_sweep(
        axs[2], dict_list, substrate_temp=PROBE_STATION_TEMP
    )
    axs[1].legend(
        loc="lower right",
        fontsize=5,
        frameon=False,
        handlelength=1,
        handleheight=1,
        borderpad=0.1,
        labelspacing=0.2,
    )
    axs[1].set_box_aspect(1.0)
    axs[2].set_box_aspect(1.0)
    axs[2].set_xlim(-500, 500)
    axs[2].xaxis.set_major_locator(MultipleLocator(250))
    return axs


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
    plot_combined_figure(axs, data_list)
    if save_combined:
        plt.subplots_adjust(wspace=0.4)
        fig3.savefig(combined_path, bbox_inches="tight")
    return (fig1, ax1), (fig2, ax2), (fig3, axs)


if __name__ == "__main__":
    run_all_dc_sweep_plots()
    plt.show()
