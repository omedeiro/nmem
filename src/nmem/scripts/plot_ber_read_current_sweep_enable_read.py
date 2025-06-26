import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_read_current_sweep_data
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import (
    plot_read_current_sweep_enable_read,
)

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    dict_list, data_list, data_list2 = import_read_current_sweep_data()
    fig, axs = plot_read_current_sweep_enable_read(dict_list, data_list, data_list2)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_enable_read.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
