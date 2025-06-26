import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size
from nmem.analysis.trace_plots import (
    plot_time_concatenated_traces,
)

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    dict_list = import_directory("../data/voltage_trace_array_emulation")

    figsize = get_consistent_figure_size("wide")
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
    plot_time_concatenated_traces(axs, dict_list[:5])

    if save_dir:
        plt.savefig(
            f"{save_dir}/voltage_trace_array_emulation.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
