#! /usr/bin/env python3
"""
Bit error rate (BER) as a function of the read current for various write current setpoints. 
The switching probability is also plotted for each read current.

The width of each operating region (defined where the trace exceeds and returns from a ±5% bound from 0.5 is plotted as a function of the write current. 
The width is approximately equal to the stored persistent current. 

Incomplete analysis. Attempting to derive a relationship between the write current and the stored persistent current. 
"""

import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_read_current_operating

# Apply global plot styling
apply_global_style()



def main(save_dir=None):
    dict_list = import_directory(
        "../data/ber_sweep_read_current/write_current/write_current_sweep_C3"
    )
    plot_read_current_operating(dict_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_operating.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
