import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    get_channel_temperature,
)
from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import (
    add_colorbar,
    plot_fill_between_array,
    plot_read_sweep_array,
    set_plot_style,
)

set_plot_style()

if __name__ == "__main__":
    data = import_directory("data")

    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    enable_read_310_C4_list = import_directory("data_310uA_C4")

    data_inverse = import_directory("data_inverse")

    dict_list = [enable_read_290_list, enable_read_300_list, enable_read_310_list]
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.5)

    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    cax = fig.add_subplot(gs[3])  # dedicated colorbar axis
    for i in range(3):
        plot_read_sweep_array(
            axs[i], dict_list[i], "bit_error_rate", "enable_read_current"
        )
        enable_write_temp = get_channel_temperature(dict_list[i][0], "write")
        plot_fill_between_array(axs[i], dict_list[i])
        axs[i].set_xlim(400, 1000)
        axs[i].set_ylim(0, 1)
        axs[i].set_xlabel("$I_{\mathrm{read}}$ [µA]")
        axs[i].set_title(
            f"$I_{{EW}}$={290 + i * 10} [µA]\n$T_{{W}}$={enable_write_temp:.2f} [K]", fontsize=8
        )
        axs[i].set_box_aspect(1.0)
        axs[i].xaxis.set_major_locator(plt.MultipleLocator(200))
    axs[0].set_ylabel("BER")
    # axs[2].legend(
    #     frameon=False,
    #     loc="upper left",
    #     bbox_to_anchor=(1, 1),
    #     title="Enable Read Current,\n Read Temperature",
    # )

    axpos = axs[2].get_position()
    cbar = add_colorbar(axs[2], dict_list, "enable_read_current", cax=cax)
    cbar.ax.set_position([axpos.x1 + 0.02, axpos.y0, 0.01, axpos.y1 - axpos.y0])
    cbar.set_ticks(plt.MaxNLocator(nbins=6))

    save_fig = False
    if save_fig:
        plt.savefig("read_current_sweep_three2.pdf", bbox_inches="tight")
