#! /usr/bin/env python3

"""
Plot Bit Error Rate (BER) Memory Retention

Generates a plot showing the relationship between memory retention time and bit error rate
Times shorter than 10ms were programmed by adjusting the waveform output. Longer delays were programmed with software defined triggers. 

BER calculated from 200e3 measurements. The bias settings were kept constant for all measurements. i.e. the settings were not reoptimized at each point. 

"""


import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_delay_data
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_retention

# Apply global plot styling
apply_global_style()



def main(save_dir=None):
    delay_list, bit_error_rate_list, _ = import_delay_data()
    plot_retention(delay_list, bit_error_rate_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_memory_retention.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
