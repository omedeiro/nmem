import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from nmem.analysis.analysis import (
    import_directory,
    plot_fill_between_array,
    plot_read_sweep_array,
)

def plot_three_panel_sweep(axs:list[plt.Axes], dict_list:list[dict[str, list[float]]]):
    for i in range(3):
        plot_read_sweep_array(
            axs[i], dict_list[i], "bit_error_rate", "enable_read_current"
        )
        plot_fill_between_array(axs[i], dict_list[i])
        axs[i].set_xlim(400, 1000)
        axs[i].set_ylim(0, 1)
        axs[i].set_xlabel("Read Current ($\mu$A)")

    axs[0].set_ylabel("Bit Error Rate")
    axs[2].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Enable Read Current,\n Read Temperature",
    )
    return axs

if __name__ == "__main__":
    # Import
    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    dict_list = [enable_read_290_list, enable_read_300_list, enable_read_310_list]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(7, 4.3), sharey=True)
    plot_three_panel_sweep(axs, dict_list)

    save = False
    if save:
        fig.savefig("read_current_sweep_three.pdf", bbox_inches="tight")
