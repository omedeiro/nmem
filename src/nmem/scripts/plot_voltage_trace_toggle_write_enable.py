import matplotlib.pyplot as plt

from nmem.analysis.constants import TRACE_INDEX
from nmem.analysis.data_import import (
    import_directory,
)
from nmem.analysis.sweep_plots import (
    plot_bit_error_rate,
)
from nmem.analysis.trace_plots import (
    plot_voltage_trace_stack,
)


def main(save_dir=None):
    dict_list = import_directory("../data/voltage_trace_toggle_write_enable")
    data_off = dict_list[0]
    data_on = dict_list[1]

    # Plots with the enable on
    fig, ax = plt.subplots()
    ax = plot_bit_error_rate(ax, data_on)
    ax.set_xlabel("Write Current [$\\mu$A]")

    if save_dir:
        plt.savefig(
            f"{save_dir}/voltage_write_current_sweep_on.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()

    fig, axs = plt.subplots(3, 1)
    ax = plot_voltage_trace_stack(axs, data_on, trace_index=TRACE_INDEX)

    if save_dir:
        plt.savefig(
            f"{save_dir}/voltage_trace_stack_on.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()

    # Plots with the enable off
    fig, ax = plt.subplots()
    ax = plot_bit_error_rate(ax, data_off)
    ax.set_xlabel("Write Current [$\\mu$A]")

    if save_dir:
        plt.savefig(
            f"{save_dir}/voltage_write_current_sweep_off.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    else:
        plt.show()

    fig, axs = plt.subplots(3, 1)
    ax = plot_voltage_trace_stack(axs, data_off, trace_index=TRACE_INDEX)

    if save_dir:
        plt.savefig(
            f"{save_dir}/voltage_trace_stack_off.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
