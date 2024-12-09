import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.ticker import MultipleLocator

from nmem.analysis.analysis import load_data, plot_threshold

font_path = r"C:\\Users\\ICE\\AppData\\Local\\Microsoft\\Windows\\Fonts\\Inter-VariableFont_opsz,wght.ttf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams["figure.figsize"] = [3.5, 2.36]
plt.rcParams["font.size"] = 5.5
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Inter"


def text_from_bit(bit: str):
    if bit == "0":
        return "WR0"
    elif bit == "1":
        return "WR1"
    elif bit == "N":
        return ""
    elif bit == "R":
        return "RD"
    elif bit == "E":
        return "ER"
    elif bit == "W":
        return "EW"
    elif bit == "z":
        return "RD0"
    elif bit == "Z":
        return "W0R1"
    elif bit == "o":
        return "RD1"
    elif bit == "O":
        return "W1R0"
    else:
        return None


def plot_data_delay_manu(data_dict_keyd):
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.2, 0.8, 3))
    data_dict = data_dict_keyd[0]
    INDEX = 14
    fig, ax = plt.subplots(figsize=(2.6, 3.54))
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.subplot(611)
    x = data_dict["trace_chan_in"][0][:, INDEX] * 1e6
    yin = np.mean(data_dict["trace_chan_in"][1], axis=1) * 1e3
    (p1,) = plt.plot(x, yin, color=colors[0], label="Input")
    plt.xticks(np.linspace(x[0], x[-1], 11), labels=None)
    plt.xlim([x[0], x[-1]])
    ax = plt.gca()
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(data_dict["bitmsg_channel"][0]):
        text = text_from_bit(bit)
        plt.text(
            i + 0.6,
            axheight * 1.1,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.tick_params(direction="in")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    plt.ylim([-150, 1100])
    plt.yticks([0, 500, 1200])
    # plt.grid(axis="x", which="both")

    plt.subplot(612)
    x = data_dict["trace_enab"][0][:, INDEX] * 1e6
    y = np.mean(data_dict["trace_enab"][1], axis=1) * 1e3
    (p2,) = plt.plot(x, y, color=colors[-1], label="Enable")
    plt.xticks(np.linspace(x[0], x[-1], 11), labels=None)
    ax = plt.gca()
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(data_dict["bitmsg_enable"][0]):
        text = text_from_bit(bit)
        plt.text(
            i + 0.5,
            axheight * 0.96,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    ax.tick_params(direction="in")
    ax.set_xticklabels([])
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.ylim([-10, 100])
    plt.xlim([x[0], x[-1]])
    plt.yticks([0, 50])
    # plt.grid(axis="x", which="both")

    plt.subplot(613)
    x = data_dict["trace_chan_out"][0][:, INDEX] * 1e6
    yout = data_dict["trace_chan_out"][1][:, INDEX] * 1e3
    # yout = np.roll(yout, 10)*1.3
    (p3,) = plt.plot(x, yout, color=colors[1], label="Output")
    # plt.grid(axis="x", which="both")
    ax = plt.gca()
    ax = plot_threshold(ax, 4, 5, 400)
    ax = plot_threshold(ax, 9, 10, 400)
    plt.xlabel("Time [$\mu$s]")
    plt.ylim([-150, 900])
    plt.xlim([0, 10])
    plt.yticks([0, 500])
    ax = plt.gca()
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate("NNNNzNNNNo"):
        text = text_from_bit(bit)
        plt.text(
            i + 0.5,
            axheight * 0.7,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    ax.tick_params(direction="in")
    ax.set_xticklabels([])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.sca(ax)
    plt.xticks(np.linspace(x[0], x[-1], 3), labels=None)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    plt.subplot(614)
    data_dict = data_dict_keyd[1]
    x = data_dict["trace_chan_out"][0][:, INDEX] * 1e6
    yout = data_dict["trace_chan_out"][1][:, INDEX] * 1e3
    # yout = np.roll(yout, 10)*1.3
    (p3,) = plt.plot(x, yout, color=colors[1], label="Output")
    # plt.grid(axis="x", which="both")
    ax = plt.gca()
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate("NNNNZNNNNO"):
        text = text_from_bit(bit)
        plt.text(
            i + 0.2,
            axheight * 0.95,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    ax = plot_threshold(ax, 4, 5, 400)
    ax = plot_threshold(ax, 9, 10, 400)
    plt.xlabel("Time [$\mu$s]")
    plt.ylim([-150, 900])
    plt.xlim([0, 10])
    plt.yticks([0, 500])
    ax = plt.gca()
    ax.tick_params(direction="in")
    ax.set_xticklabels([])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.sca(ax)
    plt.xticks(np.linspace(x[0], x[-1], 3), labels=None)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    plt.subplot(615)
    data_dict = data_dict_keyd[2]
    x = data_dict["trace_chan_out"][0][:, INDEX] * 1e6
    yout = data_dict["trace_chan_out"][1][:, INDEX] * 1e3
    # yout = np.roll(yout, 10)*1.3
    (p3,) = plt.plot(x, yout, color=colors[1], label="Output")
    # plt.grid(axis="x", which="both")
    ax = plt.gca()
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate("NNNNzNNNNO"):
        text = text_from_bit(bit)
        plt.text(
            i + 0.2,
            axheight * 1.2,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    ax = plot_threshold(ax, 4, 5, 400)
    ax = plot_threshold(ax, 9, 10, 400)
    plt.xlabel("Time [$\mu$s]")
    plt.ylim([-150, 900])
    plt.xlim([0, 10])
    plt.yticks([0, 500])
    ax = plt.gca()
    ax.tick_params(direction="in")
    ax.set_xticklabels([])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.sca(ax)
    plt.xticks(np.linspace(x[0], x[-1], 3), labels=None)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    plt.subplot(616)
    data_dict = data_dict_keyd[2]
    x = data_dict["trace_chan_out"][0][:, -1] * 1e6
    yout = data_dict["trace_chan_out"][1][:, -1] * 1e3
    # yout = np.roll(yout, 10)*1.3
    (p3,) = plt.plot(x, yout, color=colors[1], label="Output")
    # plt.grid(axis="x", which="both")
    ax = plt.gca()
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate("NNNNZNNNNo"):
        text = text_from_bit(bit)
        plt.text(
            i + 0.51,
            axheight * 1.0,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    ax = plot_threshold(ax, 4, 5, 400)
    ax = plot_threshold(ax, 9, 10, 400)
    plt.ylim([-150, 900])
    plt.xlim([0, 10])
    plt.yticks([0, 500])
    ax = plt.gca()
    ax.tick_params(direction="in")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.sca(ax)
    plt.xticks(np.linspace(x[0], x[-1], 3))

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    fig = plt.gcf()
    # fig.supylabel("Voltage [mV]", x=1, y=0.5, rotation=-90)
    # fig.supxlabel("Time [$\mu$s]", x=0.5, y=0.05)
    print(fig.get_size_inches())
    print(fig.get_dpi())
    plt.savefig("delay_manu.pdf", bbox_inches="tight")
    plt.show()
    return data_dict


INVERSE_COMPARE_DICT = {
    0: load_data(
        "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-31-23.mat"
    ),
    1: load_data(
        "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-23-55.mat"
    ),
    2: load_data(
        "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 16-04-36.mat"
    ),
}

if __name__ == "__main__":
    inverse_compare_dict = {
        0: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-31-23.mat"
        ),
        1: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-23-55.mat"
        ),
        2: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 16-04-36.mat"
        ),
    }

    plot_data_delay_manu(inverse_compare_dict)
