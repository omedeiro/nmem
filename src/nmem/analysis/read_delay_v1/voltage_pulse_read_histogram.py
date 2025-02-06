import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from nmem.analysis.analysis import (
    plot_voltage_trace_averaged,
    import_directory,
    plot_voltage_trace_bitstream,
    plot_voltage_hist,
    get_bit_error_rate,
)

plt.rcParams["figure.figsize"] = [7, 3.5]
plt.rcParams["font.size"] = 5
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.frameon"] = False


plt.rcParams["xtick.major.size"] = 1
plt.rcParams["ytick.major.size"] = 1

def plot_read_delay(ax: Axes, dict_list: list[dict]) -> Axes:
    bers = []
    for i in range(4):
        bers.append(get_bit_error_rate(dict_list[i]))

    ax.plot([1, 2, 3, 4], bers, label="bit_error_rate", marker="o", color="#345F90")
    ax.set_xlabel("Delay [$\mu$s]")
    ax.set_ylabel("BER")

    return ax


def create_trace_hist_plot(
    ax_dict: dict[str, Axes], dict_list: list[dict], save: bool = False
) -> Axes:
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()

    plot_voltage_trace_averaged(
        ax_dict["A"], dict_list[4], "trace_write_avg", color="#293689", label="Write"
    )
    plot_voltage_trace_averaged(
        ax2, dict_list[4], "trace_ewrite_avg", color="#ff1423", label="Enable Write"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"], dict_list[4], "trace_read0_avg", color="#1966ff", label="Read 0"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"],
        dict_list[4],
        "trace_read1_avg",
        color="#ff14f0",
        linestyle="--",
        label="Read 1",
    )
    plot_voltage_trace_averaged(
        ax3, dict_list[4], "trace_eread_avg", color="#ff1423", label="Enable Read"
    )

    plot_voltage_hist(ax_dict["C"], dict_list[3])

    ax_dict["A"].legend(loc="upper left")
    ax_dict["A"].set_ylabel("[mV]")
    ax2.legend()
    ax2.set_ylabel("[mV]")
    ax3.legend()
    ax3.set_ylabel("[mV]")
    ax_dict["B"].set_xlabel("Time [ns]")
    ax_dict["B"].set_ylabel("[mV]")
    ax_dict["B"].legend(loc="upper left")

    return ax_dict

if __name__ == "__main__":
    # dict_list = import_directory("data")
    # fig, ax = plt.subplots()
    # plot_read_delay(ax, dict_list)
    # plt.show()

    # fig, ax = plt.subplots()
    # plot_voltage_hist(ax, dict_list[3])
    # plt.show()

    # dict_list2 = import_directory("data2")
    # for data_dict in dict_list2:
    #     fig, ax = plt.subplots()
    #     plot_voltage_trace_bitstream(ax, data_dict, "trace_chan_out")
    #     ax.set_xlabel("Time [$\mu$s]")
    #     ax.set_ylabel("Voltage [V]")
    #     plt.show()

    fig, ax = plt.subplots(figsize=(60/25.4, 45/25.4))
    dict_list = import_directory("data")
    plot_voltage_hist(ax, dict_list[3])
    # plt.show()
    save = True
    if save:
        plt.savefig("delay_plotting_v3.pdf", bbox_inches="tight")


    fig, ax_dict = plt.subplot_mosaic("A;B", figsize=(60/25.4, 45/25.4), constrained_layout=True)
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()
    plot_voltage_trace_averaged(
        ax_dict["A"], dict_list[4], "trace_write_avg", color="#293689", label="Write"
    )
    plot_voltage_trace_averaged(
        ax2, dict_list[4], "trace_ewrite_avg", color="#ff1423", label="Enable Write"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"], dict_list[4], "trace_read0_avg", color="#1966ff", label="Read 0"
    )
    plot_voltage_trace_averaged(
        ax_dict["B"],
        dict_list[4],
        "trace_read1_avg",
        color="#ff14f0",
        linestyle="--",
        label="Read 1",
    )
    plot_voltage_trace_averaged(
        ax3, dict_list[4], "trace_eread_avg", color="#ff1423", label="Enable Read"
    )

    ax_dict["A"].legend(loc="upper left")
    ax_dict["A"].set_ylabel("[mV]")
    ax2.legend()
    ax2.set_ylabel("[mV]")
    ax3.legend()
    ax3.set_ylabel("[mV]")
    ax_dict["B"].set_xlabel("Time [ns]")
    ax_dict["B"].set_ylabel("[mV]")
    ax_dict["B"].legend(loc="upper left")
    plt.savefig("delay_plotting_v3_trace.pdf", bbox_inches="tight")