import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import (
    plot_critical_currents_from_dc_sweep,
)
from nmem.analysis.trace_plots import (
    plot_combined_dc_figure,
    plot_current_voltage_from_dc_sweep,
)

apply_global_style()


def main(data_dir="../data/dc_sweep", save_dir=None):
    """
    Main function to generate IV temperature sweep plots.
    """
    data_list = import_directory(data_dir)

    # Create individual plots
    fig1, ax1 = plt.subplots()
    plot_critical_currents_from_dc_sweep(ax1, data_list)

    fig2, ax2 = plt.subplots()
    plot_current_voltage_from_dc_sweep(ax2, data_list)

    # Create combined figure
    fig3, axs = plt.subplots(1, 3, figsize=(7, 4))
    plot_combined_dc_figure(axs, data_list)
    plt.subplots_adjust(wspace=0.4)

    if save_dir:
        # Save all figures
        fig1.savefig(
            f"{save_dir}/iv_temperature_sweep_critical_currents.png",
            bbox_inches="tight",
            dpi=300,
        )
        fig2.savefig(
            f"{save_dir}/iv_temperature_sweep_current_voltage.png",
            bbox_inches="tight",
            dpi=300,
        )
        fig3.savefig(
            f"{save_dir}/iv_temperature_sweep_combined.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
