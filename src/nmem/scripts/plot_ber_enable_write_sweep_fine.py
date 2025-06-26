#!/usr/bin/env python3
"""
Plot the bit error rate as a function of the enable write current. 

Sweeps are shown for different write currents, allowing analysis of 
how the enable write current affects the bit error rate across various 
write current set points.

"""
import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_enable_write_sweep_fine

# Apply global plot styling
apply_global_style()



def main(data_dir="../data/ber_sweep_enable_write_current/data2", save_dir=None):
    """
    Main function to plot fine enable write sweep.
    """
    data_list2 = import_directory(data_dir)
    plot_enable_write_sweep_fine(data_list2)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_enable_write_sweep_fine.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
