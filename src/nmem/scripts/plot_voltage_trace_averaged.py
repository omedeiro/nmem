#! /usr/bin/env python3

"""
Script to plot averaged voltage traces from a directory of data files.

"""

import matplotlib.pyplot as plt

from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style
from nmem.analysis.trace_plots import (
    plot_voltage_pulse_avg,
)

# Apply global plot styling
apply_global_style()


def main(save_dir=False):
    dict_list = import_directory("../data/voltage_trace_averaged")
    plot_voltage_pulse_avg(dict_list)

    if save_dir:
        plt.savefig(
            f"{save_dir}/voltage_trace_averaged.pdf",
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
