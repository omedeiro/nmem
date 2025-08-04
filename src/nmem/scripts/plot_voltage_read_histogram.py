'''
Script to plot the read amplitudes from 200e3 write and read operations. 
Voltages are binned according to the previously written state. 
'''
import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.histogram_utils import (
    plot_voltage_hist,
)
from nmem.analysis.styles import apply_global_style

# Apply global plot styling
apply_global_style()


def main(save_dir=None):
    dict_list = import_directory("../data/voltage_trace_averaged")
    fig, ax = plt.subplots()
    plot_voltage_hist(ax, dict_list[1])
    
    ax.text(
        0.5,
        0.95,
        "Cell C3",
        transform=ax.transAxes,
        fontsize=6,
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8
        ),
    )
    if save_dir:
        plt.savefig(
            f"{save_dir}/voltage_read_histogram.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
