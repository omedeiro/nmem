import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.axes import Axes

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


def text_from_bit(bit: str) -> str:
    if bit == "0":
        return "0"
    elif bit == "1":
        return "1"
    elif bit == "N":
        return ""
    elif bit == "R":
        return ""
    elif bit == "E":
        return ""
    elif bit == "W":
        return ""
    else:
        return None


def plot_message(ax: Axes, message: str) -> Axes:
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(message):
        text = text_from_bit(bit)
        ax.text(
            i + 0.5,
            axheight * 0.5,
            text,
            ha="center",
            va="center",
            fontsize=7,
            color="black",
        )

    return ax


def plot_trace_averaged(ax: Axes, data_dict: dict, trace_name: str, color: str) -> Axes:
    ax.plot(
        (data_dict[trace_name][0, :] - data_dict[trace_name][0, 0]) * 1e9,
        data_dict[trace_name][1, :],
        label=trace_name,
        color=color,
    )
    return ax


def plot_hist(ax: Axes, data_dict: dict) -> Axes:
    ax.hist(
        data_dict["read_zero_top"][0, :],
        log=True,
        range=(0.2, 0.6),
        bins=100,
        label="Read 0",
        color="#345F90",
    )
    ax.hist(
        data_dict["read_one_top"][0, :],
        log=True,
        range=(0.2, 0.6),
        bins=100,
        label="Read 1",
        color="#B3252C",
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

    create_combined_plot(data_dict, save=False)
