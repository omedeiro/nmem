
'''
Script to plot the bit error rate (BER) for read current sweeps at write currents from 10 to 15µA.

'''

import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_read_sweep_write_current

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    fig, ax = plt.subplots()
    dict_list = import_directory(
        "../data/ber_sweep_read_current/write_current/fine_sweep"
    )
    plot_read_sweep_write_current(dict_list, ax=ax)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)


    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_write_current_fine.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
