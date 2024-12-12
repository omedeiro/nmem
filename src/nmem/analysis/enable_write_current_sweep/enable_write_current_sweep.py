from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from nmem.measurement.cells import CELLS

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 12


def calculate_right_critical_currents(
    enable_write_currents: np.ndarray, cell: str, width_ratio: float, iretrap: float
) -> np.ndarray:
    channel_critical_current = (enable_write_currents * cell["slope"]) + cell[
        "intercept"
    ]
    right_critical_current = channel_critical_current * (
        1 + ((1 / width_ratio) * iretrap)
    )
    return right_critical_current


def polygon_under_graph(x: np.ndarray, y: np.ndarray) -> list:
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.0), *zip(x, y), (x[-1], 0.0)]


def plot_waterfall(ax: Axes3D, data_dict: dict) -> Axes3D:

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(data_dict)))
    verts_list = []
    zlist = []

    for key, data in data_dict.items():
        enable_write_currents = data["x"][:, :, 0].flatten() * 1e6
        current_cell = data["cell"][0]
        left_critical_currents = (
            enable_write_currents * CELLS[current_cell]["slope"]
        ) + CELLS[current_cell]["intercept"]
        ber = data["bit_error_rate"].flatten()
        write_current = data["write_current"].flatten()[0] * 1e6
        zlist.append(write_current)
        verts = polygon_under_graph(left_critical_currents, ber)
        verts_list.append(verts)

    poly = PolyCollection(verts_list, facecolors=colors, alpha=0.6, edgecolors="k")
    ax.add_collection3d(poly, zs=zlist, zdir="y")

    ax.set_xlabel("$I_{{C, H_L}}$ ($\mu$A)", labelpad=10)
    ax.set_ylabel("$I_W$ ($\mu$A)", labelpad=70)
    ax.set_zlabel("BER", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=12, pad=5)

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)

    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])
    ax.set_ylim(10, zlist[-1])
    ax.set_yticks(zlist)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_box_aspect([0.5, 1, 0.2], zoom=0.8)
    ax.view_init(20, -35)
    ax.grid(False)
    return ax


def find_operating_peaks(data_dict: dict) -> Tuple[float, float]:
    bit_error_rate: np.ndarray = data_dict["bit_error_rate"].flatten()
    enable_write_currents: np.ndarray = data_dict.get("x")[:, :, 0].flatten() * 1e6

    right_critical_currents = calculate_right_critical_currents(
        enable_write_currents, CELLS[CURRENT_CELL], WIDTH_RATIO, IRETRAP
    )
    left_critical_currents = right_critical_currents / WIDTH_RATIO

    nominal_peak = np.mean(left_critical_currents[bit_error_rate < 0.3])
    inverting_peak = np.mean(left_critical_currents[bit_error_rate > 0.7])

    return nominal_peak, inverting_peak


def find_operating_width(data_dict: dict) -> Tuple[float, float]:
    bit_error_rate: np.ndarray = data_dict["bit_error_rate"].flatten()
    enable_write_currents: np.ndarray = data_dict.get("x")[:, :, 0].flatten() * 1e6

    right_critical_currents = calculate_right_critical_currents(
        enable_write_currents, CELLS[CURRENT_CELL], WIDTH_RATIO, IRETRAP
    )
    left_critical_currents = right_critical_currents / WIDTH_RATIO
    nominal_peak = left_critical_currents[bit_error_rate < 0.4]
    nominal_width = nominal_peak[-1] - nominal_peak[0]
    inverting_peak = left_critical_currents[bit_error_rate > 0.6]
    inverting_width = inverting_peak[-1] - inverting_peak[0]

    return nominal_width, inverting_width


def get_operating_widths(data_dict: dict) -> Tuple[list, list]:
    widths = []
    write_currents = []
    for key in data_dict.keys():
        write_currents.append(data_dict[key]["write_current"].flatten()[0] * 1e6)
        nominal_width, inverting_width = find_operating_width(data_dict[key])
        widths.append((nominal_width, inverting_width))
    return write_currents, widths


def get_operating_peaks(data_dict: dict) -> Tuple[list, list]:
    peaks = []
    write_currents = []
    for key in data_dict.keys():
        write_currents.append(data_dict[key]["write_current"].flatten()[0] * 1e6)
        nominal_peak, inverting_peak = find_operating_peaks(data_dict[key])
        peaks.append((nominal_peak, inverting_peak))
    return write_currents, peaks


def plot_enable_write_sweep_single(
    ax: Axes,
    data_dict: dict,
) -> Axes:
    enable_write_currents = data_dict.get("x")[:, :, 0].flatten() * 1e6
    current_cell = data_dict.get("cell")[0]
    ber = data_dict.get("bit_error_rate").flatten()
    write_current = data_dict.get("write_current").flatten()[0] * 1e6

    right_critical_currents = calculate_right_critical_currents(
        enable_write_currents, CELLS[current_cell], WIDTH_RATIO, IRETRAP
    )
    left_critical_currents = np.array([right_critical_currents / WIDTH_RATIO]).flatten()

    ax.plot(
        left_critical_currents,
        ber,
        label=f"$I_{{W}}$ = {write_current:.1f}$\mu$A",
        marker=".",
        markeredgecolor="k",
    )
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis="x", colors="m")
    ax.set_xlim(left_critical_currents[-1], left_critical_currents[0])
    ax.set_xlabel("Left Critical Current ($\mu$A)", color="m")
    ax.set_ylabel("Bit Error Rate")
    ax.grid(True, which="both", axis="x")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")

    ax2 = ax.twiny()
    ax2.set_xlabel("Right Critical Current ($\mu$A)", color="b")
    ax2.set_xlim(
        left_critical_currents[-1] * WIDTH_RATIO,
        left_critical_currents[0] * WIDTH_RATIO,
    )
    ax2.xaxis.set_major_locator(MultipleLocator(50))
    ax2.tick_params(axis="x", colors="b")

    return ax


def plot_enable_write_sweep_multiple(ax: Axes, data_dict: dict) -> Axes:
    for key in data_dict.keys():
        plot_enable_write_sweep_single(ax, data_dict[key])

    return ax


def plot_linear_fit(ax: Axes, x: np.ndarray, y: np.ndarray) -> Axes:
    m, b = np.polyfit(x, y, 1)
    ax.plot(
        x,
        y,
        label="Peak Distance",
        marker="o",
        linestyle="None",
        markeredgecolor="k",
        markerfacecolor="None",
        markeredgewidth=2,
    )
    ax.text(
        np.mean([x[0], x[-1]]),
        np.mean([y[0], y[-1]]) * 1.1,
        f"y = {m:.2f}x + {b:.2f}",
        ha="right",
        va="bottom",
        fontsize=12,
    )
    # PLot the fit line
    x = np.array([x[0], x[-1]])
    ax.plot(x, m * x + b, "k--")
    return ax


def plot_peak_distance(ax: Axes, data_dict: dict) -> Axes:
    write_currents, peaks = get_operating_peaks(data_dict)
    peak_distances = np.array([x[0] - x[1] for x in peaks])
    ax.plot(
        write_currents,
        [x[0] - x[1] for x in peaks],
        label="Peak Distance",
        marker="o",
        color="C4",
    )
    # plot a linear fit
    plot_linear_fit(ax, write_currents[5:-1], peak_distances[5:-1])
    ax.set_xlabel("Write Current ($\mu$A)")
    ax.set_ylabel("Peak Distance ($\mu$A)")
    ax.grid(True)

    return ax


def plot_peak_width(ax: Axes, data_dict: dict) -> Axes:
    write_currents, widths = get_operating_widths(data_dict)
    ax.plot(write_currents, [x[0] for x in widths], label="Nominal Width")
    ax.plot(write_currents, [x[1] for x in widths], label="Inverting Width")
    ax.set_xlabel("Write Current ($\mu$A)")
    ax.set_ylabel("Width ($\mu$A)")
    ax.legend()
    return ax


def plot_peak_locations(ax: Axes, data_dict: dict) -> Axes:
    write_currents, peaks = get_operating_peaks(data_dict)
    ax.plot(write_currents, [x[0] for x in peaks], label="Nominal Peak")
    ax.plot(write_currents, [x[1] for x in peaks], label="Inverting Peak")
    ax.legend()
    ax.set_ylabel("Peak Location ($\mu$A)")
    ax.set_xlabel("Write Current ($\mu$A)")

    return ax


if __name__ == "__main__":
    data_dict = {
        0: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-52-33.mat"
        ),
        1: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-45-20.mat"
        ),
        2: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-33-31.mat"
        ),
        3: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-26-47.mat"
        ),
        4: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-20-06.mat"
        ),
        5: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-22-38.mat"
        ),
        6: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-41-06.mat"
        ),
        7: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-09-12.mat"
        ),
        8: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-14-52.mat"
        ),
        9: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-49-37.mat"
        ),
        10: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-41-43.mat"
        ),
        11: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-49-23.mat"
        ),
        12: sio.loadmat(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-19-05.mat"
        ),
    }

    data_dict1 = {
        0: data_dict[5],
        1: data_dict[12],
    }
    data_dict2 = {
        0: data_dict[1],
        1: data_dict[3],
        2: data_dict[5],
        3: data_dict[6],
        4: data_dict[7],
        5: data_dict[8],
        6: data_dict[9],
        7: data_dict[10],
        8: data_dict[11],
        9: data_dict[12],
    }

    ALPHA = 0.612
    WIDTH_RATIO = 1.8
    IRETRAP = 0.82
    IREAD = 630
    CURRENT_CELL = "C1"
    ICHL = 150
    ICHR = ICHL * WIDTH_RATIO

    fig, ax = plt.subplots()
    plot_enable_write_sweep_multiple(ax, data_dict1)

    fig, ax = plt.subplots()
    plot_peak_distance(ax, data_dict)

    fig, ax = plt.subplots()
    plot_peak_locations(ax, data_dict)

    fig, ax = plt.subplots()
    plot_peak_width(ax, data_dict)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16, 9))
    plot_waterfall(ax, data_dict2)
