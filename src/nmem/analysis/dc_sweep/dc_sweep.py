import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from nmem.analysis.analysis import (
    import_directory,
    plot_critical_currents_from_dc_sweep,
    plot_current_voltage_from_dc_sweep,
)
from nmem.simulation.spice_circuits.plotting import apply_snm_style

apply_snm_style()

PROBE_STATION_TEMP = 3.5
def plot_combined_figure(ax: Axes, dict_list: list, save: bool = True) -> Axes:
    ax[0].set_axis_off()
    ax[1] = plot_current_voltage_from_dc_sweep(ax[1], dict_list)
    ax[2] = plot_critical_currents_from_dc_sweep(ax[2], dict_list, substrate_temp=PROBE_STATION_TEMP)
    ax[1].legend(
        loc="lower right",
        fontsize=5,
        frameon=False,
        handlelength=1,
        handleheight=1,
        borderpad=0.1,
        labelspacing=0.2,
    )
    ax[1].set_box_aspect(1.0)
    ax[2].set_box_aspect(1.0)
    ax[2].set_xlim(-500, 500)
    ax[2].xaxis.set_major_locator(MultipleLocator(250))
    if save:
        plt.subplots_adjust(wspace=0.4)
        plt.savefig("iv_curve_combined.pdf", bbox_inches="tight")

    return ax


if __name__ == "__main__":
    data_list = import_directory("data")

    # fig, ax = plt.subplots()
    # plot_critical_currents_from_dc_sweep(ax, data_list)
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_current_voltage_from_dc_sweep(ax, data_list)
    # plt.show()

    fig, ax = plt.subplots(1, 3, figsize=(7, 4))
    plot_combined_figure(ax, data_list)

    plt.show()
