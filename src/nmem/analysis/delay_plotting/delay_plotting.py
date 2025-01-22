import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.axes import Axes

from nmem.analysis.measure_enable_response.enable_analysis import plot_all_cells
from nmem.analysis.analysis import plot_message
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




def plot_trace_averaged(ax: Axes, data_dict: dict, trace_name: str, **kwargs) -> Axes:
    ax.plot(
        (data_dict[trace_name][0, :] - data_dict[trace_name][0, 0]) * 1e9,
        data_dict[trace_name][1, :]*1e3,
        **kwargs
    )
    return ax


def plot_hist(ax: Axes, data_dict: dict) -> Axes:
    ax.hist(
        data_dict["read_zero_top"][0, :],
        log=True,
        range=(0.2, 0.6),
        bins=100,
        label="Read 0",
        color="#1966ff",
    )
    ax.hist(
        data_dict["read_one_top"][0, :],
        log=True,
        range=(0.2, 0.6),
        bins=100,
        label="Read 1",
        color="#ff1423",
    )
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Counts")
    ax.legend()
    return ax


def plot_bitstream(ax: Axes, data_dict: dict, trace_name: str) -> Axes:
    ax.plot(
        data_dict[trace_name][0, :] * 1e6,
        data_dict[trace_name][1, :],
        label=trace_name,
        color="#345F90",
    )
    plot_message(ax, data_dict["bitmsg_channel"][0])
    return ax


def plot_delay(ax: Axes, data_dict: dict) -> Axes:
    bers = []
    for i in range(4):
        bers.append(data_dict[i]["bit_error_rate"][0])

    ax.plot([1, 2, 3, 4], bers, label="bit_error_rate", marker="o", color="#345F90")
    ax.set_xlabel("Delay [$\mu$s]")
    ax.set_ylabel("BER")

    return ax


def create_combined_plot(data_dict: dict, save=False):
    fig = plt.figure(figsize=(7, 3.5))
    subfigs = fig.subfigures(
        2,
        3,
        width_ratios=[0.3, 0.7, 0.3],
        height_ratios=[0.4, 0.6],
        wspace=-0.1,
        hspace=0.0,
    )
    ax = subfigs[1, 0].subplots(4, 1)
    subfigs[1, 0].subplots_adjust(hspace=0.0)
    subfigs[1, 0].supylabel("Voltage [V]", x=-0.05)
    subfigs[1, 0].supxlabel("Time [ns]")
    plot_trace_averaged(ax[0], data_dict[4], "trace_write_avg", "#345F90")
    plot_trace_averaged(ax[1], data_dict[4], "trace_ewrite_avg", "#345F90")
    plot_trace_averaged(ax[2], data_dict[4], "trace_read0_avg", "#345F90")
    plot_trace_averaged(ax[2], data_dict[4], "trace_read1_avg", "#B3252C")
    plot_trace_averaged(ax[3], data_dict[4], "trace_eread_avg", "#345F90")
    ax[2].legend(["Read 0", "Read 1"], frameon=False)
    axhist = subfigs[0, 2].add_subplot()
    subfigs[0, 2].subplots_adjust(hspace=0.0)
    plot_hist(axhist, data_dict[3])

    axbits = subfigs[1, 1].subplots(4, 1)
    plot_bitstream(axbits[0], data_dict[5], trace_name="trace_chan_out")
    plot_bitstream(axbits[1], data_dict[6], trace_name="trace_chan_out")
    plot_bitstream(axbits[2], data_dict[7], trace_name="trace_chan_out")
    plot_bitstream(axbits[3], data_dict[8], trace_name="trace_chan_out")
    subfigs[1, 1].supylabel("Voltage [V]", x=0.05)
    subfigs[1, 1].supxlabel("Time [$\mu$s]")

    axdelay = subfigs[1, 2].add_subplot()
    plot_delay(axdelay, data_dict)

    fig.patch.set_visible(False)
    if save:
        plt.savefig("delay_plotting.pdf", bbox_inches="tight")
    plt.show()


def create_combined_plot_v2(data_dict: dict, save=False):
    fig = plt.figure(figsize=(7.087, 3.543))
    subfigs = fig.subfigures(
        2,
        3,
    )
    ax = subfigs[1, 1].subplots(2, 1)
    # subfigs[1, 1].subplots_adjust(hspace=0.0)
    subfigs[1, 1].supylabel("Voltage [V]", x=-0.05)
    # subfigs[1, 1].supxlabel("Time [ns]")
    ax2 = ax[0].twinx()
    ax3 = ax[1].twinx()
    plot_trace_averaged(ax[0], data_dict[4], "trace_write_avg", color="#293689", label="Write")
    plot_trace_averaged(ax2, data_dict[4], "trace_ewrite_avg", color="#ff1423" , label="Enable Write")
    plot_trace_averaged(ax[1], data_dict[4], "trace_read0_avg", color="#1966ff", label="Read 0")
    plot_trace_averaged(ax[1], data_dict[4], "trace_read1_avg", color="#ff14f0", linestyle="--", label="Read 1")
    plot_trace_averaged(ax3, data_dict[4], "trace_eread_avg", color="#ff1423", label="Enable Read")
    ax[1].legend(["Read 0", "Read 1"], frameon=False)
    ax[1].set_xlabel("Time [ns]")
    ax2.set_ylabel("[V]")

    axfit = subfigs[1, 2].add_subplot()
    plot_all_cells(axfit)
    fig.patch.set_visible(False)
    if save:
        plt.savefig("delay_plotting_v2.pdf", bbox_inches="tight")
    plt.show()


def create_combined_plot_v3(data_dict: dict, save=False):
    fig = plt.figure(figsize=(6.264, 2))
    ax_dict = fig.subplot_mosaic("AC;BC")    
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()
    plot_trace_averaged(ax_dict["A"], data_dict[4], "trace_write_avg", color="#293689", label="Write")
    plot_trace_averaged(ax2, data_dict[4], "trace_ewrite_avg", color="#ff1423" , label="Enable Write")
    plot_trace_averaged(ax_dict["B"], data_dict[4], "trace_read0_avg", color="#1966ff", label="Read 0")
    plot_trace_averaged(ax_dict["B"], data_dict[4], "trace_read1_avg", color="#ff14f0", linestyle="--", label="Read 1")
    plot_trace_averaged(ax3, data_dict[4], "trace_eread_avg", color="#ff1423", label="Enable Read")
    ax_dict["A"].legend(loc="upper left")
    ax_dict["A"].set_ylabel("[mV]")
    ax2.legend()
    ax2.set_ylabel("[mV]")
    ax3.legend()
    ax3.set_ylabel("[mV]")
    ax_dict["B"].set_xlabel("Time [ns]")
    ax_dict["B"].set_ylabel("[mV]")
    ax_dict["B"].legend(loc="upper left")

    fig.subplots_adjust(wspace=0.45)
    plot_all_cells(ax_dict["C"])
    if save:
        plt.savefig("delay_plotting_v2.pdf", bbox_inches="tight")
    plt.show()


def create_combined_plot_v4(data_dict: dict, save=False):
    fig = plt.figure(figsize=(6.264, 2))
    ax_dict = fig.subplot_mosaic("AC;BC")    
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()
    plot_trace_averaged(ax_dict["A"], data_dict[4], "trace_write_avg", color="#293689", label="Write")
    plot_trace_averaged(ax2, data_dict[4], "trace_ewrite_avg", color="#ff1423" , label="Enable Write")
    plot_trace_averaged(ax_dict["B"], data_dict[4], "trace_read0_avg", color="#1966ff", label="Read 0")
    plot_trace_averaged(ax_dict["B"], data_dict[4], "trace_read1_avg", color="#ff14f0", linestyle="--", label="Read 1")
    plot_trace_averaged(ax3, data_dict[4], "trace_eread_avg", color="#ff1423", label="Enable Read")
    ax_dict["A"].legend(loc="upper left")
    ax_dict["A"].set_ylabel("[mV]")
    ax2.legend()
    ax2.set_ylabel("[mV]")
    ax3.legend()
    ax3.set_ylabel("[mV]")
    ax_dict["B"].set_xlabel("Time [ns]")
    ax_dict["B"].set_ylabel("[mV]")
    ax_dict["B"].legend(loc="upper left")

    fig.subplots_adjust(wspace=0.45)
    plot_hist(ax_dict["C"], data_dict[3])

    if save:
        plt.savefig("delay_plotting_v2.pdf", bbox_inches="tight")
    plt.show()




if __name__ == "__main__":
    data_dict = {
        0: sio.loadmat(
            "SPG806_20241114_nMem_parameter_sweep_D6_A4_B4_2024-11-14 11-47-10.mat"
        ),
        1: sio.loadmat(
            "SPG806_20241114_nMem_parameter_sweep_D6_A4_B4_2024-11-14 12-01-12.mat"
        ),
        2: sio.loadmat(
            "SPG806_20241114_nMem_parameter_sweep_D6_A4_B4_2024-11-14 12-13-23.mat"
        ),
        3: sio.loadmat(
            "SPG806_20241114_nMem_parameter_sweep_D6_A4_B4_2024-11-14 12-27-15.mat"
        ),
        4: sio.loadmat(
            "SPG806_20241119_nMem_parameter_sweep_D6_A4_B4_2024-11-19 10-14-54.mat"
        ),
        5: sio.loadmat(
            "SPG806_20241119_nMem_parameter_sweep_D6_A4_B4_2024-11-19 10-01-52.mat"
        ),
        6: sio.loadmat(
            "SPG806_20241119_nMem_parameter_sweep_D6_A4_B4_2024-11-19 10-45-37.mat"
        ),
        7: sio.loadmat(
            "SPG806_20241119_nMem_parameter_sweep_D6_A4_B4_2024-11-19 10-49-50.mat"
        ),
        8: sio.loadmat(
            "SPG806_20241119_nMem_parameter_sweep_D6_A4_B4_2024-11-19 10-52-23.mat"
        ),
    }

    fig, ax = plt.subplots()
    plot_delay(ax, data_dict)
    plt.show()

    fig, ax = plt.subplots()
    plot_hist(ax, data_dict[3])
    plt.show()

    fig, ax = plt.subplots()
    plot_bitstream(ax, data_dict[5], "trace_chan_out")
    plt.show()

    create_combined_plot_v4(data_dict, save=True)
