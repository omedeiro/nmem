#!/usr/bin/env python3
"""
Plot state current vs write current relationship.

This script analyzes the relationship between state current markers and
write current, showing how different write currents affect the stored
current states in the memory cell.
"""
import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.constants import IRM
from nmem.analysis.core_analysis import (
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_read_currents,
    get_write_current,
)
from nmem.analysis.currents import get_state_current_markers
from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size

# Apply global plot styling
apply_global_style()


def main(
    data_dir="../data/ber_sweep_read_current/write_current/write_current_sweep_C3",
    save_dir=None,
):
    """
    Main function to generate state current vs write current plots.
    """
    dict_list = import_directory(data_dir)

    figsize = get_consistent_figure_size("single")
    fig, ax = plt.subplots(figsize=figsize)

    # Plot state current markers vs write current
    for data_dict in dict_list:
        state_current_markers = get_state_current_markers(data_dict, "read_current")
        write_current = get_write_current(data_dict)
        for i, state_current in enumerate(state_current_markers[0, :]):
            if state_current > 0:
                ax.plot(
                    write_current,
                    state_current,
                    label=f"{write_current} $\\mu$A",
                )

    # Get the last write current for setting limits
    write_current = get_write_current(dict_list[-1])
    ax.set_xlim(0, write_current)
    ax.set_ylabel("$I_{\\mathrm{state}}$ [$\\mu$A]")
    ax.set_xlabel("$I_{\\mathrm{write}}$ [$\\mu$A]")

    # Add boundary lines based on BER analysis
    ic_list = [IRM]
    write_current_list = [0]
    ic_list2 = [IRM]
    write_current_list2 = [0]

    for data_dict in dict_list:
        write_current = get_write_current(data_dict)
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        read_currents = get_read_currents(data_dict)

        # Ensure berargs has at least 4 elements and handle NaN values properly
        if len(berargs) >= 4:
            if not np.isnan(berargs[0]) and write_current < 100:
                arg_idx = int(berargs[0])
                if 0 <= arg_idx < len(read_currents):
                    ic_list.append(read_currents[arg_idx])
                    write_current_list.append(write_current)
            if not np.isnan(berargs[2]) and write_current > 100:
                arg_idx = int(berargs[3])
                if 0 <= arg_idx < len(read_currents):
                    ic_list.append(read_currents[arg_idx])
                    write_current_list.append(write_current)

            if not np.isnan(berargs[1]):
                arg_idx = int(berargs[1])
                if 0 <= arg_idx < len(read_currents):
                    ic_list2.append(read_currents[arg_idx])
                    write_current_list2.append(write_current)
            if not np.isnan(berargs[3]):
                arg_idx = int(berargs[2])
                if 0 <= arg_idx < len(read_currents):
                    ic_list2.append(read_currents[arg_idx])
                    write_current_list2.append(write_current)

    ax.plot(write_current_list, ic_list, "-", color="grey", linewidth=0.5)
    ax.plot(write_current_list2, ic_list2, "-", color="grey", linewidth=0.5)
    ax.set_xlim(0, 300)
    ax.set_ylabel("$I_{\\mathrm{read}}$ [$\\mu$A]")
    ax.set_xlabel("$I_{\\mathrm{write}}$ [$\\mu$A]")
    ax.axhline(IRM, color="black", linestyle="--", linewidth=0.5)

    if save_dir:
        fig.savefig(
            f"{save_dir}/state_current_vs_write_current.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
