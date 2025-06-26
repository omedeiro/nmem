import matplotlib.pyplot as plt

from nmem.analysis.core_analysis import (
    extract_temp_current_data,
    process_write_temp_arrays,
)
from nmem.analysis.data_import import (
    import_directory,
    load_and_process_write_sweep_data,
)
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.sweep_plots import (
    plot_enable_sweep,
    plot_temp_vs_current,
    plot_write_sweep_ber,
    plot_write_temp_vs_current,
)

# Apply global plot styling
apply_global_style()


def main(
    enable_write_sweep_path="../data/ber_sweep_enable_write_current/data1",
    write_sweep_path="../data/ber_sweep_write_current/enable_write",
    save_dir=None,
):
    dict_list_ews = import_directory(enable_write_sweep_path)
    dict_list_ws = load_and_process_write_sweep_data(write_sweep_path)

    figsize = get_consistent_figure_size("wide")
    fig, axs = plt.subplot_mosaic(
        "AB;CD", figsize=figsize, width_ratios=[1, 0.25], constrained_layout=True
    )

    # Panel A
    plot_enable_sweep(axs["A"], dict_list_ews, add_legend=True)

    # Panel B
    write_current_array, write_temp_array, critical_current_zero = (
        process_write_temp_arrays(dict_list_ews)
    )
    plot_write_temp_vs_current(
        axs["B"], write_current_array, write_temp_array, critical_current_zero
    )

    # Panel C
    plot_write_sweep_ber(axs["C"], dict_list_ws)

    # Panel D
    data, data2 = extract_temp_current_data(dict_list_ws)
    plot_temp_vs_current(axs["D"], data, data2)

    if save_dir:
        fig.savefig(
            f"{save_dir}/ber_write_current_sweep_operation.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
