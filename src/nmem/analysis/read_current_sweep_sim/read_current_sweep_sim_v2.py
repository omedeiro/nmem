
import ltspice
import numpy as np
from matplotlib import pyplot as plt

from nmem.analysis.core_analysis import (
    filter_first,
)
from nmem.analysis.data_import import (
    import_directory,
    import_read_current_sweep_sim_data,
    import_simulation_data,
)
from nmem.analysis.plotting import (
    CMAP,
    plot_simulation_results,
)
from nmem.simulation.spice_circuits.functions import process_read_data
from nmem.analysis.sweep_plots import plot_read_current_sweep_sim

def main():
    files, ltsp_data_dict, dict_list, write_current_list2 = (
        import_read_current_sweep_sim_data()
    )
    plot_read_current_sweep_sim(files, ltsp_data_dict, dict_list, write_current_list2)


if __name__ == "__main__":
    # Import and sort simulation files
    files = import_simulation_data("data")

    ltsp_data = ltspice.Ltspice("nmem_cell_read_example_trace.raw").parse()
    ltsp_data_dict = process_read_data(ltsp_data)

    inner = [
        ["T0", "T1", "T2", "T3"],
    ]
    innerb = [
        ["B0", "B1", "B2", "B3"],
    ]
    inner2 = [
        ["A", "B"],
    ]
    inner3 = [
        ["C", "D"],
    ]
    outer_nested_mosaic = [
        [inner],
        [innerb],
    ]
    fig, axs = plt.subplot_mosaic(
        outer_nested_mosaic,
        figsize=(6, 3),
        height_ratios=[1, 0.25],
    )

    CASE = 16
    selected_handles, selected_labels2 = plot_simulation_results(
        axs, ltsp_data_dict, case=CASE
    )
    case_current = ltsp_data_dict[CASE]["read_current"][CASE]

    dict_list = import_directory(
        "../read_current_sweep_write_current2/write_current_sweep_C3"
    )
    write_current_list = []
    for data_dict in dict_list:
        write_current = filter_first(data_dict["write_current"])
        write_current_list.append(write_current * 1e6)

    sorted_args = np.argsort(write_current_list)
    dict_list = [dict_list[i] for i in sorted_args]
    write_current_list = [write_current_list[i] for i in sorted_args]

    colors = CMAP(np.linspace(0, 1, len(dict_list)))
    col_set = [colors[i] for i in [0, 2, -1]]
    files = [files[i] for i in [0, 2, -1]]
    max_write_current = 300
    for i, file in enumerate(files):
        data = ltspice.Ltspice(f"data/{file}").parse()
        ltsp_data_dict = process_read_data(data)
        ltsp_write_current = ltsp_data_dict[0]["write_current"][0]

    axs["T1"].set_ylabel("")
    axs["T2"].set_ylabel("")
    axs["T3"].set_ylabel("")
    axs["B1"].set_ylabel("")
    axs["B2"].set_ylabel("")
    axs["B3"].set_ylabel("")

    fig.subplots_adjust(hspace=0.6, wspace=0.5)
    fig.patch.set_alpha(0)

    ax_legend = fig.add_axes([0.5, 0.95, 0.1, 0.01])
    ax_legend.axis("off")
    ax_legend.legend(
        selected_handles,
        selected_labels2,
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        handlelength=2.5,
        fontsize=8,
    )
    save_fig = False
    if save_fig:
        plt.savefig("spice_comparison_sim.pdf", bbox_inches="tight")
    plt.show()
