#!/usr/bin/env python3
"""
Plot read current operating width vs write current with persistent current.

This script analyzes the operating width of read current as a function of
write current, overlaying the theoretical persistent current relationship.
Shows the stored persistent current relationship and operating margins.

See primary sweep: plot_ber_vs_read_current_array.py
"""
import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.constants import (
    IRHL_TR,
    IRM,
)
from nmem.analysis.core_analysis import (
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_read_currents,
    get_write_current,
)
from nmem.analysis.data_import import import_directory
from nmem.analysis.styles import apply_global_style, get_consistent_figure_size

# Apply global plot styling
apply_global_style()


def main(
    data_dir="../data/ber_sweep_read_current/write_current/write_current_sweep_C3",
    save_dir=None,
):
    """
    Main function to generate read current operating width plots.
    """
    dict_list = import_directory(data_dir)

    figsize = get_consistent_figure_size("single")
    fig, ax = plt.subplots(figsize=figsize)

    # Extract operating width data from BER analysis
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

    # Calculate theoretical persistent current
    write_current_array = np.linspace(
        write_current_list[0], write_current_list[-1], 1000
    )
    persistent_current = np.where(
        write_current_array > IRHL_TR / 2,
        np.abs(write_current_array - IRHL_TR),
        write_current_array,
    )
    persistent_current = np.where(
        persistent_current > IRHL_TR, IRHL_TR, persistent_current
    )

    # Convert to arrays for calculation
    ic = np.array(ic_list)
    ic2 = np.array(ic_list2)
    delta_read_current = np.subtract(ic2, ic)

    # Plot the measured operating width
    ax.plot(
        write_current_list,
        np.abs(delta_read_current),
        label="Measured Width",
    )
    ax.set_xlabel("$I_{\\mathrm{write}}$ [$\\mu$A]")
    ax.set_ylabel("$|\\Delta I_{\\mathrm{read}}|$ [$\\mu$A]")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 110)
    ax.patch.set_alpha(0)
    ax.set_zorder(1)

    # Add twin axis for persistent current
    ax2 = ax.twinx()
    ax2.plot(
        write_current_array,
        persistent_current,
        "-",
        color="grey",
        zorder=-1,
        label="Theoretical Persistent Current",
    )
    ax2.set_ylabel("$I_{\\mathrm{persistent}}$ [$\\mu$A]")
    ax2.set_ylim(0, 110)
    ax2.set_zorder(0)
    ax2.fill_between(
        write_current_array,
        np.zeros_like(write_current_array),
        persistent_current,
        color="black",
        alpha=0.1,
    )

    # Add legends
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    if save_dir:
        fig.savefig(
            f"{save_dir}/ber_vs_read_current_array_analysis.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
