import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from nmem.analysis.currents import (
    get_channel_temperature,
    get_enable_read_current,
)
from nmem.analysis.data_import import import_directory
from nmem.analysis.plot_utils import (
    add_colorbar,
)
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import plot_read_sweep_array, plot_read_sweep_write_current

# Apply global plot styling
apply_global_style()


def plot_read_temp_sweep_C3():
    figsize = get_consistent_figure_size("wide")
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.5)

    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    cax = fig.add_subplot(gs[3])  # dedicated colorbar axis
    dict_list = [
        import_directory(
            "../data/ber_sweep_read_current/write_current/write_current_sweep_C3"
        ),
        import_directory(
            "../data/ber_sweep_read_current/write_current/write_current_sweep_C3_4"
        ),
        import_directory(
            "../data/ber_sweep_read_current/write_current/write_current_sweep_C3_3"
        ),
        # import_directory("write_current_sweep_C3_1"),
    ]
    for i, data_dict in enumerate(dict_list):
        enable_temperature = get_channel_temperature(data_dict[0], "read")
        enable_read_current = get_enable_read_current(data_dict[0])
        plot_read_sweep_array(
            axs[i],
            data_dict,
            "bit_error_rate",
            "write_current",
            marker=".",
        )
        axs[i].set_xlabel("Read Current [µA]")
        axs[i].set_ylabel("Bit Error Rate")
        axs[i].set_title(
            f"Enable Read Current: {enable_read_current} [µA]\n"
            f"T= {enable_temperature:.2f} [K]\n"
        )
        axs[i].set_box_aspect(1.0)
        axs[i].set_xlim(600, 800)
    # axs[i].legend(
    #     frameon=False,
    #     bbox_to_anchor=(1.1, 1.2),
    #     loc="upper left",
    #     title="Write Current [µA]",
    # )
    axpos = axs[2].get_position()
    cbar = add_colorbar(axs[2], dict_list, "write_current", cax=cax)
    cbar.ax.set_position([axpos.x1 + 0.02, axpos.y0, 0.01, axpos.y1 - axpos.y0])

    return fig, axs


def plot_read_sweep_import(data_dict: dict[str, list[float]]):
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, data_dict, "bit_error_rate", "write_current")
    cell = data_dict[0]["cell"][0]

    ax.set_xlabel("Read Current [µA]")
    ax.set_ylabel("Bit Error Rate")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Write Current [µA]",
    )
    ax.set_title(f"Cell {cell}")
    return fig, ax


def main(save_dir=None):
    """
    Main function to plot read current sweep data.
    """
    plot_read_sweep_write_current(import_directory("../data/ber_sweep_read_current/write_current/write_current_sweep_B2_0"))
    plot_read_sweep_write_current(import_directory("../data/ber_sweep_read_current/write_current/write_current_sweep_B2_1"))
    plot_read_sweep_write_current(import_directory("../data/ber_sweep_read_current/write_current/write_current_sweep_B2_2"))

    plot_read_sweep_write_current(import_directory("../data/ber_sweep_read_current/write_current/write_current_sweep_A2"))
    plot_read_sweep_write_current(import_directory("../data/ber_sweep_read_current/write_current/write_current_sweep_C2"))

    fig, axs = plot_read_temp_sweep_C3()

    if save_dir:
        fig.savefig(
            f"{save_dir}/ber_read_current_sweep_write_current2.png", bbox_inches="tight"
        )
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
