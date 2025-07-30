"""
# Script to plot the bit error rate (BER) for read current sweeps at 
# multiple enable read currents and three enable write currents.

"""

import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_read_current_sweep_three_data
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_read_current_sweep_three

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    dict_list = import_read_current_sweep_three_data()
    plot_read_current_sweep_three(dict_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_enable_write_trio.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
