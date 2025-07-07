#! /usr/bin/env python3

"""
Plot Bit Error Rate (BER) as a function of read current for various enable write widths.

The width of the pulse is defined by the number of points in the waveform. 
Approximately, the pulse widths range from 5ns to 150ns. 

"""
    
import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.sweep_plots import plot_read_sweep_array

# Apply global plot styling
apply_global_style()



def main(
    data_dir="../data/ber_sweep_read_current/enable_write_width",
    save_dir=None,
):
    dict_list = import_directory(data_dir)
    fig, ax = plt.subplots()
    plot_read_sweep_array(ax, dict_list, "bit_error_rate", "enable_write_width")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        title="Enable Write Width [pts]",
    )
    ax.set_yscale("log")
    ax.set_ylim([1e-3, 1])
    ax.set_xlabel("Read Current ($\\mu$A)")
    ax.set_ylabel("Bit Error Rate")

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_read_current_sweep_ew_width.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
