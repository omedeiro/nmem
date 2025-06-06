from typing import List

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from nmem.analysis.data_import import (
    import_directory,
)
from nmem.analysis.plotting import (
    plot_critical_currents_from_dc_sweep,
    plot_current_voltage_from_dc_sweep,
    set_plot_style,
)

set_plot_style()

PROBE_STATION_TEMP = 3.5
def plot_combined_figure(axs: List[Axes], dict_list: list, save: bool = False) -> List[Axes]:
    axs[0].set_axis_off()
    plot_current_voltage_from_dc_sweep(axs[1], dict_list)
    plot_critical_currents_from_dc_sweep(axs[2], dict_list, substrate_temp=PROBE_STATION_TEMP)
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
    if save:
        plt.subplots_adjust(wspace=0.4)
        plt.savefig("iv_curve_combined.pdf", bbox_inches="tight")

    return axs


if __name__ == "__main__":
    data_list = import_directory("data")

    fig, ax = plt.subplots()
    plot_critical_currents_from_dc_sweep(ax, data_list)
    plt.show()

    fig, ax = plt.subplots()
    plot_current_voltage_from_dc_sweep(ax, data_list)
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(7, 4))
    plot_combined_figure(axs, data_list)

    plt.show()
