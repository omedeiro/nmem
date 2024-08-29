import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams["figure.figsize"] = [6, 2]
plt.rcParams["font.size"] = 12


def text_from_bit(bit: str):
    if bit == "0":
        return "Write \n0"
    elif bit == "1":
        return "Write \n1"
    elif bit == "N":
        return "-"
    elif bit == "R":
        return "Read"
    elif bit == "E":
        return "Read \nEnable"
    elif bit == "W":
        return "Write \nEnable"
    else:
        return None


def plot_message(ax: plt.Axes, message: str):
    plt.sca(ax)
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(message):
        text = text_from_bit(bit)
        plt.text(i + 0.5, axheight * 1.5, text, ha="center", va="center")

    return ax


TRACE_INDEX = 11


def plot_trace_zoom(x, y, start, end):
    xzoom = x[(x > start) & (x < end)]
    yzoom = y[(x > start) & (x < end)]
    # smooth the yzoom data
    yzoom = np.convolve(yzoom, np.ones(20) / 20, mode="same")
    plt.plot(xzoom, 400 + yzoom * 10, color="red", ls="--", lw=1)
    plt.hlines(400, start, end, color="grey", ls="--", lw=1)


def plot_chan_in(ax: plt.Axes, data_dict: dict):
    plt.sca(ax)
    message = data_dict["bitmsg_channel"][0]
    x = data_dict["trace_chan_in"][0][:, TRACE_INDEX] * 1e6
    y = data_dict["trace_chan_in"][1][:, TRACE_INDEX] * 1e3
    plt.plot(x, y, color="#08519C")
    ax = plot_message(ax, message)

    plot_trace_zoom(x, y, 0.9, 2.1)
    plot_trace_zoom(x, y, 4.9, 6.1)

    plt.xticks(np.linspace(x[0], x[-1], 11))
    plt.xlabel("Time [$\mu$s]")
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])
    plt.grid(axis="x")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.box(False)
    plt.ylabel("Voltage [mV]")
    return ax


def plot_chan_out(ax: plt.Axes, data_dict: dict):
    plt.sca(ax)
    message = data_dict["bitmsg_channel"][0]
    x = data_dict["trace_chan_out"][0][:, TRACE_INDEX] * 1e6
    y = data_dict["trace_chan_out"][1][:, TRACE_INDEX] * 1e3
    plt.plot(x, y, color="#740F15")
    ax = plot_message(ax, message)

    plt.xticks(np.linspace(x[0], x[-1], 11))
    plt.xlabel("Time [$\mu$s]")
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])
    plt.grid(axis="x")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.box(False)
    plt.ylabel("Voltage [mV]")
    return ax


def plot_enable(ax: plt.Axes, data_dict: dict):
    plt.sca(ax)
    message = data_dict["bitmsg_enable"][0]
    x = data_dict["trace_enab"][0][:, TRACE_INDEX] * 1e6
    y = data_dict["trace_enab"][1][:, TRACE_INDEX] * 1e3
    plt.plot(x, y, color="#DBB40C")
    # ax = plot_message(ax, message)

    plt.xticks(np.linspace(x[0], x[-1], 11))
    plt.xlabel("Time [$\mu$s]")
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])
    plt.grid(axis="x")

    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    plt.box(False)
    plt.ylabel("Voltage [mV]")
    return ax


def plot_measurement(ax: plt.Axes, data_dict: dict):
    plt.sca(ax)
    num_meas = data_dict["num_meas"][0][0]
    w1r0 = data_dict["write_1_read_0"][0].flatten() / num_meas
    w0r1 = data_dict["write_0_read_1"][0].flatten() / num_meas
    z = w1r0 + w0r1
    line_width = 2
    plt.plot(
        data_dict["y"][0][:, 1] * 1e6,
        w0r1,
        color="#DBB40C",
        linewidth=line_width,
        label="Write 0 Read 1",
        marker=".",
    )
    plt.plot(
        data_dict["y"][0][:, 1] * 1e6,
        w1r0,
        color="#740F15",
        linewidth=line_width,
        label="Write 1 Read 0",
        marker=".",
    )
    plt.plot(
        data_dict["y"][0][:, 1] * 1e6,
        z,
        color="#08519C",
        linewidth=line_width,
        label="Total",
        marker=".",
    )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.ylabel("Bit Error Rate")
    plt.xlabel("Write Current [$\mu$A]")
    return ax


# def plot_trace_stack(ax: plt.Axes, data_dict: dict):
#     plt.sca(ax)

#     x = data_dict["trace_chan_in"][0][:, TRACE_INDEX] * 1e6
#     y = data_dict["trace_chan_in"][1][:, TRACE_INDEX] * 1e3
#     plt.plot(x, y, color="#08519C", zdir='z', zs=0)


#     x = data_dict["trace_chan_out"][0][:, TRACE_INDEX] * 1e6
#     y = data_dict["trace_chan_out"][1][:, TRACE_INDEX] * 1e3
#     plt.plot(x, y, color="#740F15", zdir='z', zs=.1)

#     ax.set_box_aspect((6, 1, 1))
#     # ax.autoscale(enable=False)
#     # ax.autoscale_view(tight=False)
#     ax.view_init(elev=80, azim=-90, roll=0)
#     ax.set_zticks([])
#     # ax.set_position([0, 0, 6, 1])
#     return ax


def plot_trace_stack(ax, data_dict: dict):
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.subplot(311)
    x = data_dict["trace_chan_in"][0][:, TRACE_INDEX] * 1e6
    y = data_dict["trace_chan_in"][1][:, TRACE_INDEX] * 1e3
    (p1,) = plt.plot(x, y, color="#08519C", label="Input")
    plt.xticks(np.linspace(x[0], x[-1], 11), labels=None)
    # plot_trace_zoom(x, y, 2.9, 4.1)
    # plot_trace_zoom(x, y, 6.9, 8.1)

    plot_trace_zoom(x, y, 0.9, 2.1)
    plot_trace_zoom(x, y, 4.9, 6.1)

    ax = plt.gca()
    ax.set_xticklabels([])
    plt.ylim([-150, 900])
    plt.yticks([0, 500])
    plt.grid(axis="x")

    plt.subplot(312)
    x = data_dict["trace_enab"][0][:, TRACE_INDEX] * 1e6
    y = data_dict["trace_enab"][1][:, TRACE_INDEX] * 1e3
    (p2,) = plt.plot(x, y, color="#DBB40C", label="Enable")
    plt.xticks(np.linspace(x[0], x[-1], 11))
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.ylim([-10, 100])
    plt.yticks([0, 50])
    plt.grid(axis="x")

    plt.subplot(313)
    x = data_dict["trace_chan_out"][0][:, TRACE_INDEX] * 1e6
    y = data_dict["trace_chan_out"][1][:, TRACE_INDEX] * 1e3

    (p3,) = plt.plot(x, y, color="#740F15", label="Output")
    plt.grid(axis="x")
    plt.xlabel("Time [$\mu$s]")
    plt.ylim([-150, 800])
    plt.yticks([0, 500])
    # plt.hlines(data_dict["threshold_bert"].flatten()[TRACE_INDEX]*1e3, x[0], x[-1], color="red", ls="-", lw=1)
    ax = plt.gca()
    plt.sca(ax)
    plt.xticks(np.linspace(x[0], x[-1], 11))
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])

    plt.legend(
        [p1, p2, p3],
        ["Input", "Enable", "Output"],
        loc="center",
        bbox_to_anchor=(0.5, -1.5),
        ncol=3,
        frameon=False,
    )
    fig = plt.gcf()
    fig.supylabel("Voltage [mV]")

    return ax


if __name__ == "__main__":
    data_on = sio.loadmat(
        "SPG806_20240829_nMem_parameter_sweep_D6_A4_C1_2024-08-29 09-39-26.mat"
    )

    data_off = sio.loadmat(
        "SPG806_20240829_nMem_parameter_sweep_D6_A4_C1_2024-08-29 09-47-45.mat"
    )

    fig, ax = plt.subplots()
    ax = plot_measurement(ax, data_on)
    plt.show()

    fig, ax = plt.subplots()
    ax = plot_trace_stack(ax, data_on)
    plt.show()

    fig, ax = plt.subplots()
    ax = plot_measurement(ax, data_off)
    plt.show()

    fig, ax = plt.subplots()
    ax = plot_trace_stack(ax, data_off)
    plt.show()
