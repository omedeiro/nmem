#!/usr/bin/env python3
"""
Plot the bit error rate as a function of the enable write current. 

For a write amplitude of 0uA. No operation is visible. 
For write amplitudes of 5-10uA the minimum BER decreases and the width of the operating region increases.

A maximum operating region is achieved at 40uA, where the minimum BER is still near 0. 

At 50uA the minimum BER increases again, but the width of the operating region is still large.

"""
import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_enable_write_sweep_multiple

# Apply global plot styling
apply_global_style()



def main(data_dir="../data/ber_sweep_enable_write_current/data2", save_dir=None):
    """
    Main function to plot fine enable write sweep.
    """
    data_list2 = import_directory(data_dir)
    plot_enable_write_sweep_multiple(data_list2)

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
