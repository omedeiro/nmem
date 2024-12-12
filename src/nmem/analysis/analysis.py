import os
from typing import List, Literal, Tuple

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from nmem.calculations.calculations import (
    calculate_read_currents,
)
from nmem.measurement.cells import CELLS


def import_directory(file_path: str) -> list:
    data_list = []
    files = get_file_names(file_path)
    for file in files:
        data = sio.loadmat(os.path.join(file_path, file))
        data_list.append(data)
    return data_list


def get_file_names(file_path: str) -> list:
    files = os.listdir(file_path)
    files = [file for file in files if file.endswith(".mat")]
    return files


def text_from_bit(bit: str) -> str:
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


def text_from_bit_v2(bit: str):
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


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.00), *zip(x, y), (x[-1], 0.00)]


def polygon_nominal(x: np.ndarray, y: np.ndarray) -> list:
    y = np.copy(y)
    y[y > 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def polygon_inverting(x: np.ndarray, y: np.ndarray) -> list:
    y = np.copy(y)
    y[y < 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def plot_threshold(ax: Axes, start: int, end: int, threshold: float) -> Axes:
    ax.hlines(threshold, start, end, color="red", ls="-", lw=1)
    return ax


def plot_message(ax: Axes, message: str) -> Axes:
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(message):
        text = text_from_bit(bit)
        ax.text(i + 0.5, axheight * 1.45, text, ha="center", va="center", fontsize=14)

    return ax


def plot_trace_zoom(
    ax: Axes, x: np.ndarray, y: np.ndarray, start: float, end: float
) -> Axes:
    xzoom = x[(x > start) & (x < end)]
    yzoom = y[(x > start) & (x < end)]

    # smooth the yzoom data
    yzoom = np.convolve(yzoom, np.ones(20) / 20, mode="same")
    ax.plot(xzoom, 400 + yzoom * 10, color="red", ls="--", lw=1)
    ax.hlines(400, start, end, color="grey", ls="--", lw=1)

    return ax


def plot_chan_in(ax: Axes, data_dict: dict, trace_index: int) -> Axes:
    message = data_dict["bitmsg_channel"][0]
    x = data_dict["trace_chan_in"][0][:, trace_index] * 1e6
    y = data_dict["trace_chan_in"][1][:, trace_index] * 1e3
    ax.plot(x, y, color="#08519C")
    ax = plot_message(ax, message)

    plot_trace_zoom(x, y, 0.9, 2.1)
    plot_trace_zoom(x, y, 4.9, 6.1)

    ax.set_xticks(np.linspace(x[0], x[-1], 11))
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.set_xlabel("Time [$\mu$s]")
    ax.set_ylabel("Voltage [mV]")

    ax.grid(axis="x")
    return ax


def plot_chan_out(ax: Axes, data_dict: dict, trace_index: int) -> Axes:
    message = data_dict["bitmsg_channel"][0]
    x = data_dict["trace_chan_out"][0][:, trace_index] * 1e6
    y = data_dict["trace_chan_out"][1][:, trace_index] * 1e3
    ax.plot(x, y, color="#740F15")
    ax = plot_message(ax, message)

    plot_trace_zoom(x, y, 0.9, 2.1)
    plot_trace_zoom(x, y, 4.9, 6.1)

    ax.set_xticks(np.linspace(x[0], x[-1], 11))
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.set_xlabel("Time [$\mu$s]")
    ax.set_ylabel("Voltage [mV]")

    ax.grid(axis="x")
    return ax


def plot_enable(ax: Axes, data_dict: dict, trace_index: int):
    x = data_dict["trace_enab"][0][:, trace_index] * 1e6
    y = data_dict["trace_enab"][1][:, trace_index] * 1e3
    ax.plot(x, y, color="#DBB40C")

    ax.set_xticks(np.linspace(x[0], x[-1], 11))
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])
    ax.grid(axis="x")

    ax.set_xlabel("Time [$\mu$s]")
    ax.set_ylabel("Voltage [mV]")
    return ax


def find_state_currents(data_dict: dict) -> Tuple[float, float]:
    read_currents = data_dict["y"][:, :, 0].flatten() * 1e6
    ber = data_dict["bit_error_rate"].flatten()

    if np.max(ber) > 0.6:
        max_ber = np.max(ber)
        ber_threshold = 0.5 + (max_ber - 0.5) / 2
        state1_current = read_currents[ber > ber_threshold][0]
        state0_current = read_currents[ber > ber_threshold][-1]

    else:
        min_ber = np.min(ber)
        ber_threshold = 0.5 - (0.5 - min_ber) / 2
        state0_current = read_currents[ber < ber_threshold][0]
        state1_current = read_currents[ber < ber_threshold][-1]

    return state0_current, state1_current


def plot_state_current_markers(ax: Axes, data_dict: dict, **kwargs) -> Axes:
    read_currents = data_dict.get("y")[:, :, 0].flatten() * 1e6
    ber = data_dict.get("bit_error_rate").flatten()
    state0_current, state1_current = find_state_currents(data_dict)
    ax.plot(
        read_currents[read_currents == state0_current],
        ber[read_currents == state0_current],
        marker="D",
        markeredgecolor="k",
        linewidth=1.5,
        label="_state0",
        **kwargs,
    )
    ax.plot(
        read_currents[read_currents == state1_current],
        ber[read_currents == state1_current],
        marker="P",
        markeredgecolor="k",
        linewidth=1.5,
        label="_state1",
        **kwargs,
    )

    return ax


def plot_read_sweep(
    ax: Axes,
    data_dict: dict,
    value_name: Literal["bit_error_rate", "write_0_read_1", "write_1_read_0"],
    variable_name: Literal[
        "enable_write_current", "read_width", "write_width", "write_current"
    ],
    state_markers: bool = False,
    **kwargs,
) -> Axes:
    read_currents = data_dict.get("y")[:, :, 0] * 1e6
    value = data_dict.get(value_name).flatten()
    variable = data_dict.get(variable_name).flatten()[0]

    if read_currents.shape != value.shape:
        read_currents = read_currents.flatten()

    ax.plot(
        read_currents,
        value,
        label=f"{variable}",
        marker=".",
        markeredgecolor="k",
        **kwargs,
    )

    if state_markers:
        plot_state_current_markers(ax, data_dict, markersize=15, **kwargs)

    ax.set_ylim(0, 1)
    ax.set_title(f"{variable_name}")
    ax.legend(frameon=True, loc=2)
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Normalized Bit Error Rate")
    return ax


def plot_read_sweep_array(
    ax: Axes, data_dict: dict, value_name: str, variable_name: str
) -> Axes:
    for key in data_dict.keys():
        plot_read_sweep(ax, data_dict[key], value_name, variable_name)

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    return ax


def plot_read_sweep_array_3d(ax: Axes3D, data_dict: dict) -> Axes3D:
    for key in data_dict.keys():
        ax = plot_read_sweep_3d(ax, data_dict[key])

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)

    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlim(500, 950)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect([0.5, 1, 0.25], zoom=0.8)
    ax.view_init(20, -35)
    ax.grid(False)

    ax.set_position([0.0, 0.0, 1, 1])

    for child in ax.get_children():
        if isinstance(child, Rectangle):
            child.set_visible(False)
    return ax


def plot_read_sweep_3d(ax: Axes3D, data_dict: dict) -> Axes3D:

    cmap = plt.get_cmap("RdBu")
    colors = cmap(np.linspace(0, 1, 5))

    read_currents = data_dict.get("y")[:, :, 0].flatten() * 1e6
    ber = data_dict.get("bit_error_rate").flatten()
    enable_read_current = data_dict.get("enable_read_current").flatten()[0] * 1e6

    verts = polygon_nominal(read_currents, ber)
    inv_verts = polygon_inverting(read_currents, ber)

    poly = PolyCollection([verts], facecolors=colors[-1], alpha=0.6, edgecolors=None)
    poly_inv = PolyCollection(
        [inv_verts], facecolors=colors[0], alpha=0.6, edgecolors=None
    )
    ax.add_collection3d(poly, zs=[enable_read_current], zdir="y")
    ax.add_collection3d(poly_inv, zs=[enable_read_current], zdir="y")

    ax.set_xlabel("$I_R$ ($\mu$A)", labelpad=-6)
    ax.set_ylabel("$I_{{ER}}$ ($\mu$A)", labelpad=4)
    ax.set_zlabel("BER", labelpad=-6)
    ax.tick_params(axis="both", which="major", pad=-1)

    return ax


def plot_bit_error_rate(ax: Axes, data_dict: dict) -> Axes:
    num_meas = data_dict.get("num_meas")[0][0]
    w1r0 = data_dict.get("write_1_read_0")[0].flatten() / num_meas
    w0r1 = data_dict.get("write_0_read_1")[0].flatten() / num_meas
    ber = (w1r0 + w0r1) / 2
    measurement_param = data_dict.get("y")[0][:, 1] * 1e6

    ax.plot(
        measurement_param,
        w0r1,
        color="#DBB40C",
        label="Write 0 Read 1",
        marker=".",
    )
    ax.plot(
        measurement_param,
        w1r0,
        color="#740F15",
        label="Write 1 Read 0",
        marker=".",
    )
    ax.plot(
        measurement_param,
        ber,
        color="#08519C",
        label="Total Bit Error Rate",
        marker=".",
    )
    ax.legend()
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Normalized\nBit Error Rate")
    return ax


def plot_voltage_trace(
    ax: Axes, time: np.ndarray, voltage: np.ndarray, **kwargs
) -> Axes:
    ax.plot(time, voltage, **kwargs)
    ax.set_xlim(time[0], time[-1])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xticklabels([])
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="x", direction="in", which="both")
    ax.grid(axis="x", which="both")
    return ax


def plot_trace_stack_1D(
    axes: List[Axes], data_dict: dict, trace_index: int = 0
) -> List[Axes]:
    if len(axes) != 3:
        raise ValueError("The number of axes must be 3.")

    chan_in_x = data_dict.get("trace_chan_in")[0] * 1e6
    chan_in_y = data_dict.get("trace_chan_in")[1] * 1e3
    chan_out_x = data_dict.get("trace_chan_out")[0] * 1e6
    chan_out_y = data_dict.get("trace_chan_out")[1] * 1e3
    enab_in_x = data_dict.get("trace_enab")[0] * 1e6
    enab_in_y = data_dict.get("trace_enab")[1] * 1e3

    if chan_in_x.ndim == 2:
        chan_in_x = chan_in_x[:, trace_index]
        chan_in_y = chan_in_y[:, trace_index]
        chan_out_x = chan_out_x[:, trace_index]
        chan_out_y = chan_out_y[:, trace_index]
        enab_in_x = enab_in_x[:, trace_index]
        enab_in_y = enab_in_y[:, trace_index]

    bitmsg_channel = data_dict.get("bitmsg_channel")[0]
    bitmsg_enable = data_dict.get("bitmsg_enable")[0]

    ax = axes[0]
    plot_voltage_trace(ax, chan_in_x, chan_in_y, color="C0", label="Input")
    ax.legend(loc="upper left")

    if bitmsg_enable[1] == "W" and bitmsg_channel[1] != "N":
        plot_trace_zoom(ax, chan_in_x, chan_in_y, 0.9, 2.1)
        plot_trace_zoom(ax, chan_in_x, chan_in_y, 4.9, 6.1)

    if bitmsg_enable[3] == "W" and bitmsg_channel[3] != "N":
        plot_trace_zoom(ax, chan_in_x, chan_in_y, 2.9, 4.1)
        plot_trace_zoom(ax, chan_in_x, chan_in_y, 6.9, 8.1)

    ax = axes[1]
    plot_voltage_trace(ax, enab_in_x, enab_in_y, color="C1", label="Enable")
    ax.legend(loc="upper left")

    ax = axes[2]
    plot_voltage_trace(ax, chan_out_x, chan_out_y, color="C2", label="Output")
    ax.legend(loc="upper left")

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlim([0, 10])

    fig = plt.gcf()
    fig.supylabel("Voltage [mV]")
    fig.supxlabel("Time [$\mu$s]")
    fig.subplots_adjust(hspace=0.0)

    return axes


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


def plot_voltage_hist(ax: Axes, voltage: np.ndarray, **kwargs) -> Axes:
    ax.hist(voltage, bins=77, **kwargs)

    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylabel("Counts")
    ax.set_xlabel("Voltage [mV]")

    return ax


def convert_location_to_coordinates(location: str) -> tuple:
    """Converts a location like 'A1' to coordinates (x, y)."""
    column_letter = location[0]
    row_number = int(location[1:]) - 1
    column_number = ord(column_letter) - ord("A")
    return column_number, row_number


def plot_text_labels(
    ax: Axes, xloc: np.ndarray, yloc: np.ndarray, ztotal: np.ndarray, log: bool
) -> Axes:
    for x, y in zip(xloc, yloc):
        text = f"{ztotal[y, x]:.2f}"
        txt_color = "black"
        if ztotal[y, x] > (0.8 * max(ztotal.flatten())):
            txt_color = "white"
        if log:
            text = f"{ztotal[y, x]:.1e}"
            txt_color = "black"

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
    ax: Axes,
    xloc: np.ndarray,
    yloc: np.ndarray,
    ztotal: np.ndarray,
    title: str = None,
    log: bool = False,
    reverse: bool = False,
    cmap: plt.cm = None,
) -> Axes:
    if cmap is None:
        cmap = plt.get_cmap("viridis")
    if reverse:
        cmap = cmap.reversed()

    if log:
        ax.matshow(ztotal, cmap=cmap, norm=LogNorm(vmin=1e-6, vmax=1e-2))
    else:
        ax.matshow(ztotal, cmap=cmap)

    if title:
        ax.set_title(title)

    ax.set_xticks(range(4), ["A", "B", "C", "D"])
    ax.set_yticks(range(4), ["1", "2", "3", "4"])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="both", length=0)
    ax = plot_text_labels(ax, xloc, yloc, ztotal, log)

    return ax


def plot_normalization(
    ax: Axes,
    write_current_norm: np.ndarray,
    read_current_norm: np.ndarray,
    enable_write_current: np.ndarray,
    enable_read_current: np.ndarray,
) -> Axes:
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
    ax.set_xticks(rotation=45)
    ax.set_ylabel("Normalized Current")
    ax.set_yticks(np.linspace(0, 1, 11))
    return ax


def plot_analytical(ax: Axes, data_dict: dict) -> Axes:
    color_map = plt.get_cmap("RdBu")

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
    return ax


def plot_cell_param(ax: Axes, param: str) -> Axes:
    param_array = np.array([CELLS[cell][param] for cell in CELLS]).reshape(4, 4)

    plot_array(
        ax,
        np.arange(4),
        np.arange(4),
        param_array * 1e6,
        f"Cell {param}",
        log=False,
        norm=False,
        reverse=False,
    )
    return ax

def plot_fit(ax: Axes, xfit: np.ndarray, yfit: np.ndarray) -> Axes:
    z = np.polyfit(xfit, yfit, 1)
    p = np.poly1d(z)
    ax.scatter(xfit, yfit, color="#08519C")
    xplot = np.linspace(min(xfit), max(xfit), 10)
    ax.plot(xplot, p(xplot), "--", color="#740F15")
    ax.text(
        0.1,
        0.1,
        f"{p[1]:.3f}x + {p[0]:.3f}",
        fontsize=12,
        color="red",
        backgroundcolor="white",
        transform=plt.gca().transAxes,
    )

    return ax

def construct_array(data_dict: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = data_dict.get("x")[0][:, 1] * 1e6
    y = data_dict.get("y")[0][:, 0] * 1e6
    w0r1 = 100 - data_dict.get("write_0_read_1")[0].flatten()
    w1r0 = data_dict.get("write_1_read_0")[0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")
    return x, y, ztotal

def plot_enable_current_relation(ax: Axes, data_dict: dict) -> Axes:
    
    x, y, ztotal = construct_array(data_dict)
    dx, dy = np.diff(x)[0], np.diff(y)[0]
    xfit, yfit = get_fitting_points(x, y, ztotal)

    ax.scatter(xfit, yfit)

    # Plot a fit line to the scatter points
    plot_fit(ax, xfit, yfit)

    ax.matshow(
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
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_locator(MaxNLocator(len(x)))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_major_locator(MultipleLocator(50))    

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


def get_fitting_points(
    x: np.ndarray, y: np.ndarray, ztotal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    mid_idx = np.where(ztotal > np.nanmax(ztotal, axis=0) / 2)
    xfit, xfit_idx = np.unique(x[mid_idx[1]], return_index=True)
    yfit = y[mid_idx[0]][xfit_idx]
    return xfit, yfit


def plot_slice(ax: Axes, data_dict: dict) -> Axes:
    w0r1 = 100 - data_dict.get("write_0_read_1")[0].flatten()
    w1r0 = data_dict.get("write_1_read_0")[0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape(
        (data_dict["y"][0].shape[0], data_dict["x"][0].shape[0]),
        order="F",
    )
    x = data_dict.get("y")[0][:, 0] * 1e6
    ax.plot(
        x,
        ztotal,
    )

    return ax
