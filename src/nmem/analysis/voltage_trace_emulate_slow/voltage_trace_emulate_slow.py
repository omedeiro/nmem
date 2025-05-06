from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from nmem.analysis.analysis import (
    CMAP,
    get_voltage_trace_data,
    import_directory,
    plot_voltage_trace,
    set_plot_style,
)

set_plot_style()


def extract_shifted_traces(
    data_dict: dict, trace_index: int = 0, time_shift: float = 0.0
) -> Tuple:
    chan_in_x, chan_in_y = get_voltage_trace_data(
        data_dict, "trace_chan_in", trace_index
    )
    chan_out_x, chan_out_y = get_voltage_trace_data(
        data_dict, "trace_chan_out", trace_index
    )
    enab_in_x, enab_in_y = get_voltage_trace_data(data_dict, "trace_enab", trace_index)

    # Shift all x values
    chan_in_x = chan_in_x + time_shift
    chan_out_x = chan_out_x + time_shift
    enab_in_x = enab_in_x + time_shift

    return chan_in_x, chan_in_y, enab_in_x, enab_in_y, chan_out_x, chan_out_y


def plot_time_concatenated_traces(axs: List[Axes], dict_list: List[dict]) -> List[Axes]:
    colors = CMAP(np.linspace(0.1, 1, len(dict_list)))
    colors = np.flipud(colors)

    for idx, data_dict in enumerate(dict_list):
        shift = 10 * idx  # Shift time window by 10 µs per dataset
        chan_in_x, chan_in_y, enab_in_x, enab_in_y, chan_out_x, chan_out_y = (
            extract_shifted_traces(data_dict, time_shift=shift)
        )

        plot_voltage_trace(axs[0], chan_in_x, chan_in_y, color=colors[0])
        plot_voltage_trace(axs[1], enab_in_x, enab_in_y, color=colors[1])
        plot_voltage_trace(axs[2], chan_out_x, chan_out_y, color=colors[-1])

    axs[2].xaxis.set_major_locator(MultipleLocator(10))
    axs[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    axs[2].set_xlim(0, 50)
    axs[0].legend(
        ["input"],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    axs[1].legend(
        ["enable"],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    axs[2].legend(
        ["output"],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    fig = plt.gcf()
    fig.supylabel("Voltage [mV]", fontsize=9)
    fig.supxlabel("Time [µs]", y=-0.02, fontsize=9)
    fig.subplots_adjust(hspace=0.0)

    return axs


if __name__ == "__main__":
    dict_list = import_directory("data")

    fig, axs = plt.subplots(3, 1, figsize=(6, 3), sharex=True)
    plot_time_concatenated_traces(axs, dict_list[:5])
    plt.savefig("voltage_trace_emulate_slow.pdf", bbox_inches="tight")
    plt.show()
