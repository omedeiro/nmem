import os

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

from nmem.calculations.calculations import (
    calculate_persistent_current,
    calculate_read_currents,
)
from nmem.measurement.cells import CELLS


def load_data(file_path: str):
    data = sio.loadmat(file_path)
    return data


def import_directory(file_path: str):
    data_list = []
    files = get_file_names(file_path)
    for file in files:
        data = load_data(os.path.join(file_path, file))
        data_list.append(data)
    return data_list


def get_file_names(file_path: str):
    files = os.listdir(file_path)
    files = [file for file in files if file.endswith(".mat")]
    return files


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
        return "Read \nEnable"
    elif bit == "W":
        return "Write \nEnable"
    else:
        return None


def find_edge(data: np.ndarray) -> list:
    pos_data = np.argwhere(data > 0.55)
    neg_data = np.argwhere(data < 0.45)

    if len(pos_data) > 0:
        pos_edge1 = pos_data[0][0]
        neg_edge1 = pos_data[-1][0]
    else:
        pos_edge1 = 0
        neg_edge1 = 0
    if len(neg_data) > 0:
        neg_edge2 = neg_data[0][0]
        pos_edge2 = neg_data[-1][0]
    else:
        neg_edge2 = 0
        pos_edge2 = 0
    return [pos_edge1, neg_edge1, neg_edge2, pos_edge2]


def find_edge_dual(ber: np.ndarray, w0r1: np.ndarray, w1r0: np.ndarray) -> list:
    w0r1_peak = np.argmax(np.diff(w0r1))
    w0r1_peak_neg = np.argmin(np.diff(w0r1))
    w1r0_peak = np.argmax(np.diff(w1r0))
    w1r0_peak_neg = np.argmin(np.diff(w1r0))
    left_nominal = np.argmin(np.diff(w1r0))
    right_nominal = np.argmax(np.diff(w0r1))
    left_inverting = np.argmax(np.diff(w0r1))

    # if right_nominal == 0:
    #     right_nominal = np.argmax(np.diff(w1r0[left_nominal:]))

    # if right_nominal < left_nominal:
    #     right_nominal = np.argmax(np.diff(w0r1[left_nominal:]))

    # if left_nominal > right_nominal:
    #     left_nominal = np.argmin(np.diff(w1r0[:left_nominal]))

    # if right_nominal < left_nominal:
    #     right_nominal = np.argmax(np.diff(w0r1[left_nominal:]))

    while ber[left_inverting] < 0.5:
        left_inverting = np.argmax(np.diff(w0r1[:left_nominal]))

    if np.min(ber) >= 0.45:
        left_nominal = 0
        right_nominal = 0
    # if left_nominal > right_nominal:
    #     left_nominal = np.argmin(np.diff(w1r0[:left_nominal]))
    #     right_nominal = np.argmax(np.diff(ber[left_nominal:]))
    # right_nominal = np.argmax(np.diff(w0r1[right_nominal:])

    # if ber[w0r1_peak] < 0.45:
    #     w0r1_peak = np.argmax(np.diff(w0r1[:w0r1_peak]))
    # if w0r1_peak_neg > w0r1_peak:
    #     w0r1_peak = np.argmax(np.diff(w0r1[w0r1_peak_neg:]))
    # if w1r0_peak_neg > w1r0_peak:
    #     w1r0_peak_neg = np.argmin(np.diff(w1r0[:w1r0_peak_neg]))
    print(f"left_nominal: {left_nominal}, right_nominal: {right_nominal}")
    print(f"left invert: {left_inverting}")
    return [left_inverting]


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.00), *zip(x, y), (x[-1], 0.00)]


def polygon_nominal(x, y):
    y = np.copy(y)
    y[y > 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def polygon_inverting(x, y):
    y = np.copy(y)
    y[y < 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def plot_threshold(ax: plt.Axes, start, end, threshold):
    plt.sca(ax)
    plt.hlines(threshold, start, end, color="red", ls="-", lw=1)
    return ax


def plot_message(ax: plt.Axes, message: str):
    plt.sca(ax)
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(message):
        text = text_from_bit(bit)
        plt.text(i + 0.5, axheight * 1.45, text, ha="center", va="center", fontsize=14)

    return ax


def plot_trace_zoom(x, y, start, end):
    xzoom = x[(x > start) & (x < end)]
    yzoom = y[(x > start) & (x < end)]

    # smooth the yzoom data
    yzoom = np.convolve(yzoom, np.ones(20) / 20, mode="same")
    plt.plot(xzoom, 400 + yzoom * 10, color="red", ls="--", lw=1)
    plt.hlines(400, start, end, color="grey", ls="--", lw=1)


def plot_chan_in(ax: plt.Axes, data_dict: dict, trace_index: int):
    plt.sca(ax)
    message = data_dict["bitmsg_channel"][0]
    x = data_dict["trace_chan_in"][0][:, trace_index] * 1e6
    y = data_dict["trace_chan_in"][1][:, trace_index] * 1e3
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


def plot_chan_out(ax: plt.Axes, data_dict: dict, trace_index: int):
    plt.sca(ax)
    message = data_dict["bitmsg_channel"][0]
    x = data_dict["trace_chan_out"][0][:, trace_index] * 1e6
    y = data_dict["trace_chan_out"][1][:, trace_index] * 1e3
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


def plot_enable(ax: plt.Axes, data_dict: dict, trace_index: int):
    plt.sca(ax)
    x = data_dict["trace_enab"][0][:, trace_index] * 1e6
    y = data_dict["trace_enab"][1][:, trace_index] * 1e3
    plt.plot(x, y, color="#DBB40C")

    plt.xticks(np.linspace(x[0], x[-1], 11))
    plt.xlabel("Time [$\mu$s]")
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])
    plt.grid(axis="x")

    plt.box(False)
    plt.ylabel("Voltage [mV]")
    return ax


def plot_measurement(ax: plt.Axes, data_dict: dict):
    plt.sca(ax)
    num_meas = data_dict["num_meas"][0][0]
    w1r0 = data_dict["write_1_read_0"][0].flatten() / num_meas
    w0r1 = data_dict["write_0_read_1"][0].flatten() / num_meas
    z = (w1r0 + w0r1) / 2
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
    plt.yscale("log")
    plt.yticks([5e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    plt.ylim([5e-5, 1e-2])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.grid(axis="y")
    plt.ylabel("Bit Error Rate")
    plt.xlabel("Read Current [$\mu$A]")
    return ax


def plot_measurement_coarse(ax: plt.Axes, data_dict: dict):
    plt.sca(ax)
    num_meas = data_dict["num_meas"][0][0]
    w1r0 = data_dict["write_1_read_0"][0].flatten() / num_meas
    w0r1 = data_dict["write_0_read_1"][0].flatten() / num_meas
    z = (w1r0 + w0r1) / 2
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
    plt.yticks([0, 0.5, 1])
    plt.ylim([0, 1])
    plt.ylabel("Normalized\nBit Error Rate")
    plt.xlabel("Write Current [$\mu$A]")
    return ax


def plot_trace_stack_write(ax, data_dict: dict, trace_index: int, legend: bool = False):
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.subplot(311)
    x = data_dict["trace_chan_in"][0][:, trace_index] * 1e6
    y = data_dict["trace_chan_in"][1][:, trace_index] * 1e3
    (p1,) = plt.plot(x, y, color="#08519C", label="Input")
    plt.xticks(np.linspace(x[0], x[-1], 11), labels=None)

    plot_trace_zoom(x, y, 0.9, 2.1)
    plot_trace_zoom(x, y, 4.9, 6.1)

    ax = plt.gca()
    ax.set_xticklabels([])
    plt.ylim([-150, 900])
    plt.yticks([0, 500])
    plt.grid(axis="x")

    plt.subplot(312)
    x = data_dict["trace_enab"][0][:, trace_index] * 1e6
    y = data_dict["trace_enab"][1][:, trace_index] * 1e3
    (p2,) = plt.plot(x, y, color="#DBB40C", label="Enable")
    plt.xticks(np.linspace(x[0], x[-1], 11))
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.ylim([-10, 100])
    plt.yticks([0, 50])
    plt.grid(axis="x")

    plt.subplot(313)
    x = data_dict["trace_chan_out"][0][:, trace_index] * 1e6
    y = data_dict["trace_chan_out"][1][:, trace_index] * 1e3

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
    if legend:
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


def plot_trace_stack_read(ax, data_dict: dict, trace_index: int):
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.subplot(311)
    x = data_dict["trace_chan_in"][0][:, trace_index] * 1e6
    y = data_dict["trace_chan_in"][1][:, trace_index] * 1e3
    (p1,) = plt.plot(x, y, color="#08519C", label="Input")
    plt.xticks(np.linspace(x[0], x[-1], 11), labels=None)

    plot_message(plt.gca(), data_dict["bitmsg_channel"][0])

    if data_dict["bitmsg_channel"][0][1] == "0":
        plot_trace_zoom(x, y, 0.9, 2.1)
        plot_trace_zoom(x, y, 4.9, 6.1)

    if data_dict["bitmsg_channel"][0][3] == "1":
        plot_trace_zoom(x, y, 2.9, 4.1)
        plot_trace_zoom(x, y, 6.9, 8.1)

    ax = plt.gca()
    ax.set_xticklabels([])
    plt.ylim([-150, 900])
    plt.yticks([0, 500])
    plt.grid(axis="x")

    plt.subplot(312)
    x = data_dict["trace_enab"][0][:, trace_index] * 1e6
    y = data_dict["trace_enab"][1][:, trace_index] * 1e3
    (p2,) = plt.plot(x, y, color="#DBB40C", label="Enable")
    plt.xticks(np.linspace(x[0], x[-1], 11))
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.ylim([-10, 100])
    plt.yticks([0, 50])
    plt.grid(axis="x")

    plt.subplot(313)
    x = data_dict["trace_chan_out"][0][:, trace_index] * 1e6
    y = data_dict["trace_chan_out"][1][:, trace_index] * 1e3

    (p3,) = plt.plot(x, y, color="#740F15", label="Output")
    plt.grid(axis="x")
    plt.xlabel("Time [$\mu$s]")
    plt.ylim([-150, 800])
    plt.yticks([0, 500])
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


def plot_trace_stack_1D(ax, data_dict: dict):
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.subplot(311)
    x = data_dict["trace_chan_in"][0] * 1e6
    y = data_dict["trace_chan_in"][1] * 1e3
    (p1,) = plt.plot(x, y, color="#08519C", label="Input")
    plt.xticks(np.linspace(x[0], x[-1], 11), labels=None)

    plot_message(plt.gca(), data_dict["bitmsg_channel"][0])

    if (
        data_dict["bitmsg_enable"][0][1] == "W"
        and data_dict["bitmsg_channel"][0][1] != "N"
    ):
        plot_trace_zoom(x, y, 0.9, 2.1)
        plot_trace_zoom(x, y, 4.9, 6.1)

    if (
        data_dict["bitmsg_enable"][0][3] == "W"
        and data_dict["bitmsg_channel"][0][3] != "N"
    ):
        plot_trace_zoom(x, y, 2.9, 4.1)
        plot_trace_zoom(x, y, 6.9, 8.1)

    ax = plt.gca()
    ax.set_xticklabels([])
    plt.ylim([-150, 900])
    plt.yticks([0, 500])
    plt.grid(axis="x")

    plt.subplot(312)
    x = data_dict["trace_enab"][0] * 1e6
    y = data_dict["trace_enab"][1] * 1e3
    (p2,) = plt.plot(x, y, color="#DBB40C", label="Enable")
    plt.xticks(np.linspace(x[0], x[-1], 11))
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.ylim([-10, 100])
    plt.yticks([0, 50])
    plt.grid(axis="x")

    plt.subplot(313)
    x = data_dict["trace_chan_out"][0] * 1e6
    y = data_dict["trace_chan_out"][1] * 1e3

    (p3,) = plt.plot(x, y, color="#740F15", label="Output")
    plt.grid(axis="x")
    ax = plt.gca()
    ax = plot_threshold(ax, 4, 5, 360)
    ax = plot_threshold(ax, 8, 9, 360)
    plt.xlabel("Time [$\mu$s]")
    plt.ylim([-150, 800])
    plt.yticks([0, 500])
    ax = plt.gca()
    plt.sca(ax)
    plt.xticks(np.linspace(x[0], x[-1], 11))
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])
    fig = plt.gcf()
    fig.supylabel("Voltage [mV]")

    return ax


def plot_hist_2axis(data_dict: dict, trace_index: int):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    w1r0 = data_dict["read_zero_top"][0][:, trace_index]
    w0r1 = data_dict["read_one_top"][0][:, trace_index]

    ax1.hist(w1r0, bins=100, alpha=1, color="#740F15", label="Read 1")
    ax1.hist(w0r1, bins=100, alpha=0.5, color="#DBB40C", label="Read 0")
    ax2.hist(w1r0, bins=100, alpha=1, color="#740F15", label="Read 1")
    ax2.hist(w0r1, bins=100, alpha=0.5, color="#DBB40C", label="Read 0")

    ax2.set_ylim([0, 10])
    ax2.set_yticks([0, 5, 10])

    ax1.set_ylim([10, 10000])

    fig.supylabel("Count")
    fig.supxlabel("Voltage [mV]")
    plt.show()


def plot_hist(data_dict: dict, trace_index: int):
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 3))

    print(f"Bit error rate: {data_dict['bit_error_rate'][0][:, trace_index][0]:.2e}")
    w0r1 = data_dict["read_zero_top"][0][:, trace_index] * 1e3
    w1r0 = data_dict["read_one_top"][0][:, trace_index] * 1e3

    ax.hist(w1r0, bins=77, alpha=1, color="#740F15", label="Read 1")
    ax2.hist(w0r1, bins=77, alpha=0.5, color="#DBB40C", label="Read 0")

    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.set_ylim([0.1, 10000])
    ax.set_ylabel("Counts")

    ax2.set_yscale("log")
    ax2.set_yticks([1, 10, 100, 1000, 10000])
    ax2.set_ylim([0.1, 10000])

    plt.ylabel("Counts")
    plt.xlabel("Voltage [mV]")

    plt.show()


def convert_location_to_coordinates(location):
    """Converts a location like 'A1' to coordinates (x, y)."""
    column_letter = location[0]
    row_number = int(location[1:]) - 1
    column_number = ord(column_letter) - ord("A")
    return column_number, row_number


def plot_text_labels(xloc, yloc, ztotal, log=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for x, y in zip(xloc, yloc):
        text = f"{ztotal[y, x]:.2f}"
        if log:
            text = f"{ztotal[y, x]:.1e}"
        txt_color = "black"
        if ztotal[y, x] < 0.5 * max(ztotal.flatten()):
            txt_color = "white"

        ax.text(
            x,
            y,
            text,
            color=txt_color,
            backgroundcolor="none",
            ha="center",
            va="center",
            weight="bold",
        )

    return ax


def plot_array(
    xloc, yloc, ztotal, title=None, log=False, norm=False, reverse=False, ax=None
):
    if ax is None:
        fig, ax = plt.subplots()

    cmap = plt.cm.get_cmap("viridis")
    if reverse:
        cmap = plt.cm.get_cmap("viridis").reversed()

    if norm:
        im = ax.imshow(ztotal, cmap=cmap, vmin=0, vmax=1)
    else:
        im = ax.imshow(ztotal, cmap=cmap)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(range(4), ["A", "B", "C", "D"])
    ax.set_yticks(range(4), ["1", "2", "3", "4"])

    ax = plot_text_labels(xloc, yloc, ztotal, log, ax=ax)

    return ax


def plot_normalization(
    write_current_norm: np.ndarray,
    read_current_norm: np.ndarray,
    enable_write_current: np.ndarray,
    enable_read_current: np.ndarray,
):
    # remove NaN from arrays
    write_current_norm = write_current_norm[~np.isnan(write_current_norm)]
    read_current_norm = read_current_norm[~np.isnan(read_current_norm)]
    enable_write_current = enable_write_current[~np.isnan(enable_write_current)]
    enable_read_current = enable_read_current[~np.isnan(enable_read_current)]

    # remove zeros from arrays
    write_current_norm = write_current_norm[write_current_norm != 0]
    read_current_norm = read_current_norm[read_current_norm != 0]
    enable_write_current = enable_write_current[enable_write_current != 0]
    enable_read_current = enable_read_current[enable_read_current != 0]

    fig, ax = plt.subplots()
    ax.boxplot(write_current_norm.flatten(), positions=[0], widths=0.5)
    ax.boxplot(read_current_norm.flatten(), positions=[1], widths=0.5)
    ax.boxplot(enable_write_current.flatten(), positions=[2], widths=0.5)
    ax.boxplot(enable_read_current.flatten(), positions=[3], widths=0.5)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        [
            "Write Current",
            "Read Current",
            "Enable\nWrite Current",
            "Enable\nRead Current",
        ]
    )
    ax.set_xlabel("Input Type")
    plt.xticks(rotation=45)
    ax.set_ylabel("Normalized Current")
    ax.set_yticks(np.linspace(0, 1, 11))
    # plt.grid(axis="y")
    plt.show()


def plot_analytical(data_dict: dict, persistent_current=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    color_map = plt.get_cmap("RdBu")
    persistent_currents, regions = calculate_persistent_current(data_dict)
    data_dict["regions"] = regions
    if persistent_current == 0:
        data_dict["persistent_currents"] = np.zeros_like(persistent_currents)
    else:
        data_dict["persistent_currents"] = (
            np.ones_like(persistent_currents) * persistent_current
        )

    read_current_dict = calculate_read_currents(data_dict)
    inv_region = np.where(
        np.maximum(
            data_dict["read_currents_mesh"]
            - read_current_dict["one_state_currents_inv"],
            read_current_dict["zero_state_currents_inv"]
            - data_dict["read_currents_mesh"],
        )
        <= 0,
        data_dict["read_currents_mesh"],
        np.nan,
    )
    nominal_region = np.where(
        np.maximum(
            data_dict["read_currents_mesh"] - read_current_dict["zero_state_currents"],
            read_current_dict["one_state_currents"] - data_dict["read_currents_mesh"],
        )
        <= 0,
        data_dict["read_currents_mesh"],
        np.nan,
    )
    ax.pcolormesh(
        data_dict["right_critical_currents_mesh"],
        data_dict["read_currents_mesh"],
        nominal_region,
        cmap=color_map,
        vmin=-1000,
        vmax=1000,
        zorder=0,
    )
    ax.pcolormesh(
        data_dict["right_critical_currents_mesh"],
        data_dict["read_currents_mesh"],
        -1 * inv_region,
        cmap=color_map,
        vmin=-1000,
        vmax=1000,
        zorder=0,
    )
    read_current_dict["nominal"] = nominal_region
    read_current_dict["inverting"] = inv_region
    return ax, read_current_dict


def plot_cell_param(param: str, ax: plt.Axes = None):
    if ax is None:
        fig, ax = plt.subplots()

    param_array = np.array([CELLS[cell][param] for cell in CELLS]).reshape(4, 4)

    plot_array(
        np.arange(4),
        np.arange(4),
        param_array * 1e6,
        f"Cell {param}",
        log=False,
        norm=False,
        reverse=False,
    )
    return ax


def find_peak(data_dict: dict):
    x = data_dict["x"][0][:, 1] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6

    w0r1 = 100 - data_dict["write_0_read_1"][0].flatten()
    w1r0 = data_dict["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal, axis=0)

    xfit, yfit = find_enable_relation(data_dict)
    # xfit = xfit[:-8]
    # yfit = yfit[:-8]

    plt.scatter(xfit, yfit)

    # Plot a fit line to the scatter points
    plot_fit(xfit, yfit)

    plt.imshow(
        ztotal,
        extent=[
            (-0.5 * dx + x[0]),
            (0.5 * dx + x[-1]),
            (-0.5 * dy + y[0]),
            (0.5 * dy + y[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    plt.xticks(np.linspace(x[0], x[-1], len(x)))
    plt.yticks(np.linspace(y[0], y[-1], len(y)))
    plt.xlabel("Enable Current [$\mu$A]")
    plt.ylabel("Channel Current [$\mu$A]")
    plt.title(f"Cell {data_dict['cell']}")
    cbar = plt.colorbar()
    cbar.set_label("Counts")
    return ztotal


def find_max_critical_current(data):
    x = data["x"][0][:, 1] * 1e6
    y = data["y"][0][:, 0] * 1e6
    w0r1 = 100 - data["write_0_read_1"][0].flatten()
    w1r0 = data["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")
    ztotal = ztotal[:, 1]

    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal)
    mid_idx = np.where(diff == np.max(diff))

    return np.mean(y[mid_idx])


def find_enable_relation(data_dict: dict):
    x = data_dict["x"][0][:, 1] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6

    w0r1 = 100 - data_dict["write_0_read_1"][0].flatten()
    w1r0 = data_dict["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    # Find the maximum critical current using np.diff
    # diff = np.diff(ztotal, axis=0)
    # diff_fit = np.where(diff == 0, np.nan, diff)
    # mid_idx = np.where(diff_fit == np.nanmax(diff_fit, axis=0))

    # Find the maximum critical current using total counts
    mid_idx = np.where(ztotal >= np.nanmax(ztotal - 5, axis=0))
    xfit, xfit_idx = np.unique(x[mid_idx[1]], return_index=True)
    yfit = y[mid_idx[0]][xfit_idx]
    return xfit, yfit


def plot_fit(xfit, yfit):
    z = np.polyfit(xfit, yfit, 1)
    p = np.poly1d(z)
    plt.scatter(xfit, yfit, color="#08519C")
    xplot = np.linspace(190, 310, 10)
    plt.plot(xplot, p(xplot), "--", color="#740F15")
    plt.text(
        0.1,
        0.9,
        f"{p[1]:.3f}x + {p[0]:.3f}",
        fontsize=12,
        color="red",
        backgroundcolor="white",
        transform=plt.gca().transAxes,
    )



def plot_enable_current_relation(data_dict: dict):
    x = data_dict["x"][0][:, 1] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6

    w0r1 = 100 - data_dict["write_0_read_1"][0].flatten()
    w1r0 = data_dict["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal, axis=0)

    xfit, yfit = find_enable_relation(data_dict)

    plt.scatter(xfit, yfit)

    # Plot a fit line to the scatter points
    plot_fit(xfit, yfit)

    plt.imshow(
        ztotal,
        extent=[
            (-0.5 * dx + x[0]),
            (0.5 * dx + x[-1]),
            (-0.5 * dy + y[0]),
            (0.5 * dy + y[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    # plt.xticks(np.linspace(200, 300, 3))
    # plt.yticks(np.linspace(300, 900, 3))
    plt.xlim(180, 320)
    plt.ylim(280, 920)
    plt.xlabel("Enable Current [$\mu$A]")
    plt.ylabel("Channel Current [$\mu$A]")
    plt.title("Enable Current Relation")
    plt.grid()
    plt.show()

    return ztotal




def find_peak(data_dict: dict):
    x = data_dict["x"][0][:, 1] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6

    w0r1 = data_dict["write_0_read_1"][0].flatten()
    w1r0 = 100-data_dict["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xfit, yfit = find_enable_relation(data_dict)


    plt.scatter(xfit, yfit)
    # Plot a fit line to the scatter points
    plot_fit(xfit, yfit)


    plt.imshow(
        ztotal,
        extent=[
            (-0.5 * dx + x[0]),
            (0.5 * dx + x[-1]),
            (-0.5 * dy + y[0]),
            (0.5 * dy + y[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    plt.xticks(np.linspace(x[0], x[-1], len(x)))
    plt.yticks(np.linspace(y[0], y[-1], len(y)))
    plt.xlabel("Enable Current [$\mu$A]")
    plt.ylabel("Channel Current [$\mu$A]")
    plt.title(f"Cell {data_dict['cell']}")
    cbar = plt.colorbar()
    cbar.set_label("Counts")
    return ztotal


def find_max_critical_current(data):
    x = data["x"][0][:, 1] * 1e6
    y = data["y"][0][:, 0] * 1e6
    w0r1 = data["write_0_read_1"][0].flatten()
    w1r0 = 100 - data["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")
    ztotal = ztotal[:, 1]

    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal)
    mid_idx = np.where(diff == np.max(diff))

    return np.mean(y[mid_idx])


def find_enable_relation(data_dict: dict):
    x = data_dict["x"][0][:, 0] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6

    w0r1 = data_dict["write_0_read_1"][0].flatten()
    w1r0 = 100 - data_dict["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    # Find the maximum critical current using total counts
    mid_idx = np.where(ztotal > np.nanmax(ztotal, axis=0)/2)
    xfit, xfit_idx = np.unique(x[mid_idx[1]], return_index=True)
    print(xfit)
    yfit = y[mid_idx[0]][xfit_idx]


    return xfit, yfit


def plot_fit(xfit, yfit):
    z = np.polyfit(xfit, yfit, 1)
    p = np.poly1d(z)
    plt.scatter(xfit, yfit)
    xplot = np.linspace(190, 410, 10)
    plt.plot(xplot, p(xplot), "r--")
    plt.text(
        0.1,
        0.9,
        f"{p[1]:.3f}x + {p[0]:.3f}",
        fontsize=12,
        color="red",
        backgroundcolor="white",
        transform=plt.gca().transAxes,
    )


def plot_slice(data_dict: dict):
    w0r1 = 100 - data_dict["write_0_read_1"][0].flatten()
    w1r0 = data_dict["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape(
        (data_dict["y"][0].shape[0], data_dict["x"][0].shape[0]),
        order="F",
    )
    cmap = plt.cm.viridis(np.linspace(0, 1, ztotal.shape[1]))
    for enable_current_index in range(ztotal.shape[1]):
        enable_current = data_dict["x"][0][enable_current_index, 1] * 1e6
        plt.plot(
            data_dict["y"][0][:, 0] * 1e6,
            ztotal[:, enable_current_index],
            label=f"enable current = {enable_current:.2f} $\mu$A",
            color=cmap[enable_current_index],
        )

