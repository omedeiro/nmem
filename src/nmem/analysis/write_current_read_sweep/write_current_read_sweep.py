import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks

from nmem.calculations.calculations import (
    calculate_persistent_current,
    calculate_read_currents,
)
from nmem.measurement.cells import CELLS

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


CURRENT_CELL = "C1"


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


def plot_waterfall(data_dict: dict, ax: Axes3D = None):
    if ax is None:
        fig, ax = plt.subplots(projection="3d")
    plt.sca(ax)
    cmap = plt.get_cmap("viridis")
    # cmap = cmap.reversed()
    colors = cmap(np.linspace(0.2, 0.8, len(data_dict)))
    verts_list = []
    inv_verts_list = []
    zlist = []
    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        current_cell = data["cell"][0]
        ber = data["bit_error_rate"].flatten()
        write_current = data["write_current"].flatten()[0] * 1e6
        zlist.append(write_current)
        verts = polygon_nominal(read_currents, ber)
        inv_verts = polygon_inverting(read_currents, ber)
        verts_list.append(verts)
        inv_verts_list.append(inv_verts)

    poly = PolyCollection(verts_list, facecolors=colors[-1], alpha=0.6, edgecolors="k")
    poly_inv = PolyCollection(
        inv_verts_list, facecolors=colors[0], alpha=0.6, edgecolors="k"
    )
    ax.add_collection3d(poly, zs=zlist, zdir="y")
    ax.add_collection3d(poly_inv, zs=zlist, zdir="y")

    # ax.set_xlabel("$I_{{EW}}$ ($\mu$A)", labelpad=10)
    ax.set_xlabel("$I_R$ ($\mu$A)", labelpad=10)
    ax.set_ylabel("$I_{{W}}$ ($\mu$A)", labelpad=70)
    ax.set_zlabel("BER", labelpad=10)
    # ax.set_zscale("log")
    ax.tick_params(axis="both", which="major", labelsize=12, pad=5)

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)

    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlim(400, 650)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_ylim(zlist[0], zlist[-1])
    ax.set_yticks(zlist)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect([0.5, 1, 0.2], zoom=0.8)
    ax.view_init(20, -35)
    ax.grid(False)

    for key, data in data_dict.items():
        cmap = plt.get_cmap("viridis")
        colors = cmap(np.linspace(0.2, 0.8, 4))
        # colors = ["red", "darkred", "lightblue", "blue"]
        edge_dict = get_edges({key: data})
        edge_list = edge_dict[key]["edges"]
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        current_cell = data["cell"][0]
        ber = data["bit_error_rate"].flatten()
        w0r1 = data["write_0_read_1_norm"].flatten()
        w1r0 = data["write_1_read_0_norm"].flatten()
        enable_read_current = data["enable_read_current"].flatten()[0] * 1e6

        ax = plot_edge_3D(edge_dict, edge_list, key, colors, ax)
        peaks = edge_dict[key]["peaks"]
        peak_properties = edge_dict[key]["peak_properties"]
        inv_peaks = edge_dict[key]["inv_peaks"]
        inv_peak_properties = edge_dict[key]["inv_peak_properties"]
        ax.plot(
            read_currents,
            enable_read_current * np.ones_like(read_currents),
            ber,
            "-",
            color="k",
        )

        # ax.plot(
        #     read_currents,
        #     enable_read_current * np.ones_like(read_currents),
        #     w0r1,
        #     "-",
        #     color="r",
        # )
        # ax.plot(
        #     read_currents,
        #     enable_read_current * np.ones_like(read_currents),
        #     w1r0,
        #     "-",
        #     color="b",
        # )
    ax = plt.gca()
    return ax


def load_data(file_path: str):
    data = sio.loadmat(file_path)
    return data


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


def get_edges(data_dict: dict) -> dict:
    edges = {}
    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        w0r1 = data["write_0_read_1_norm"].flatten()
        w1r0 = data["write_1_read_0_norm"].flatten()
        enable_read_current = data["enable_read_current"].flatten()[0] * 1e6
        edges[key] = {}
        edges[key]["edges"] = find_edge(ber)
        # edges[key]["edges"] = find_edge_dual(ber, w0r1, w1r0)
        edges[key]["param"] = enable_read_current
        edges[key]["read_currents"] = read_currents
        peaks, peak_properties = find_peaks(
            ber, height=0.4, prominence=0.05, width=1, distance=1
        )
        inv_peaks, inv_peak_properties = find_peaks(
            1 - ber, height=0.4, prominence=0.05, width=1, distance=1
        )
        edges[key]["ber"] = ber
        edges[key]["peaks"] = peaks
        edges[key]["peak_properties"] = peak_properties
        edges[key]["inv_peaks"] = inv_peaks
        edges[key]["inv_peak_properties"] = inv_peak_properties

    return edges


def plot_edges(
    data_dict: dict, ax: plt.Axes = None, fit=True, fit_dict: dict = None
) -> plt.Axes:
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, 4))
    edge_dict = get_edges(data_dict)
    edge_list = []
    param_list = []
    if ax is None:
        fig, ax = plt.subplots()

    for key, data in edge_dict.items():
        read_current = data["read_currents"]
        param = data["param"]
        param = param * CELLS[CURRENT_CELL]["slope"] + CELLS[CURRENT_CELL]["intercept"]
        param_list.append(param)
        edge = data["edges"]
        current_edge = np.array([read_current[e] for e in edge])
        current_edge = np.where(np.array(edge) == 0, np.nan, current_edge)

        edge_list.append(current_edge)

        for i, e in enumerate(edge):
            if e == 0:
                continue
            ax.scatter(
                param, read_current[e], color=colors[i], marker="o", edgecolor="k"
            )

    if fit:
        for i in range(4):
            x = np.array(param_list)
            y = np.array([edge[i] for edge in edge_list])
            fity = y[~np.isnan(y)]
            fitx = x[~np.isnan(y)]
            fit_start = fit_dict[i]["fit_start"]
            fit_stop = fit_dict[i]["fit_stop"]
            fitx = fitx[fit_start : len(fitx) - fit_stop]
            fity = fity[fit_start : len(fity) - fit_stop]
            fit = np.polyfit(fitx, fity, 1)
            fit_fn = np.poly1d(fit)
            ax.plot(x, fit_fn(x), "--", color=colors[i])
            ax.plot(fitx, fity, "o", color="k", fillstyle="none", mew=1.5)
            ax.text(
                1.50,
                0.5 + 0.1 * i,
                f"y = {fit_fn[1]:.3f}x + {fit_fn[0]:.1f}",
                transform=ax.transAxes,
                color=colors[i],
            )

    return ax


def plot_edge_3D(edge_dict: dict, edge_list: list, key: int, colors: list, ax: Axes3D):
    for edge in edge_list:
        if edge == 0:
            continue
        ax.plot(
            [
                edge_dict[key]["read_currents"][edge],
                edge_dict[key]["read_currents"][edge],
            ],
            [edge_dict[key]["param"], edge_dict[key]["param"]],
            [0, edge_dict[key]["ber"][edge]],
            color="k",
            linestyle=":",
        )
        ax.plot(
            [
                edge_dict[key]["read_currents"][edge],
                edge_dict[key]["read_currents"][edge],
            ],
            [edge_dict[key]["param"], edge_dict[key]["param"]],
            [edge_dict[key]["ber"][edge]],
            color=colors[edge_list.index(edge)],
            marker="o",
            markeredgecolor="k",
        )

    return ax


def plot_enable_read_current_edges(
    data_dict: dict,
    analytical_data_dict: dict,
    persistent_current=None,
    fitting_dict=None,
):
    fig, ax = plt.subplots()
    ax = plot_edges(data_dict, ax, fit=False, fit_dict=fitting_dict)

    enable_write_current = data_dict[0]["enable_write_current"].flatten()[0] * 1e6

    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    write_current = 30.0
    plt.text(
        1.5,
        0.8,
        f"$I_{{EW}}$ = {enable_write_current:.1f}$\mu$A\n$I_{{W}}$ = {write_current}$\mu$A",
        transform=ax.transAxes,
    )
    ax, read_current_dict = plot_analytical(
        analytical_data_dict, persistent_current=persistent_current, ax=ax
    )

    width_ratio = WIDTH_RIGHT / WIDTH_LEFT
    retrap_gap = read_current_dict["retrap_gap"]
    retrap_difference = read_current_dict["retrap_difference"]
    ax.text(
        1.2,
        0.0,
        f"""
        Width Ratio: {width_ratio:.2f}
        Alpha: {ALPHA:.3f}
        Retrap ratio: {IRETRAP_ENABLE:.3f}
        Retrap gap: {retrap_gap:.3f}
        Retrap difference: {retrap_difference:.3f}""",
        transform=ax.transAxes,
    )

    plt.ylim(500, 950)
    plt.xlim(600, 950)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    plt.xlabel("Channel Critical Current ($\mu$A)")
    plt.ylabel("Read Current ($\mu$A)")
    plt.grid(True, which="both")
    ax.set_aspect("equal")
    plt.show()


def plot_enable_read_current_edges_stack(
    data_dict: dict,
    analytical_data_dict: dict,
    ax: plt.Axes = None,
    persistent_current=None,
    fitting_dict=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    ax = plot_edges(data_dict, ax, fit=False, fit_dict=fitting_dict)

    enable_write_current = data_dict[0]["enable_write_current"].flatten()[0] * 1e6

    ax.set_xlim(600, 950)
    ax.set_ylim(500, 950)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(100))

    ax, read_current_dict = plot_analytical(
        analytical_data_dict, persistent_current=persistent_current, ax=ax
    )

    # ax.set_ylim(500, 950)
    # ax.set_xlim(600, 950)

    # plt.title("Enable Read Current Edges")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # plt.xlabel("Channel Critical Current ($\mu$A)")
    # plt.ylabel("Read Current ($\mu$A)")
    # plt.grid(True, which="both")
    ax.set_aspect("equal")
    return ax


def plot_stack(
    data_dict_list,
    analytical_data_dict_list,
    persistent_currents_list,
    fitting_dict_list,
):
    fig, axs = plt.subplots(3, 1, figsize=(3, 9), sharex=True)
    fig.subplots_adjust(hspace=0)
    for i in range(3):
        axs[i] = plot_enable_read_current_edges_stack(
            data_dict_list[i],
            analytical_data_dict_list[i],
            axs[i],
            persistent_current=persistent_currents_list[i],
            fitting_dict=fitting_dict_list[i],
        )

    fig.supxlabel("$I_{{CH}}$ ($\mu$A)", x=0.5, y=0.04)
    fig.supylabel("$I_{{R}}$ ($\mu$A)", x=0.99, y=0.5)

    caxis = fig.add_axes([0.21, 0.9, 0.61, 0.02])
    cbar = fig.colorbar(
        axs[0].collections[-1], cax=caxis, orientation="horizontal", pad=0.1
    )
    caxis.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)
    plt.savefig("enable_read_current_edges_stack.pdf", bbox_inches="tight")
    plt.show()


def plot_read_sweep_single(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.8, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        write_current = data["write_current"].flatten()[0] * 1e6
        plt.plot(
            read_currents,
            ber,
            label=f"$I_{{W}}$ = {write_current:.1f}$\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )

    ax = plt.gca()
    plt.xlabel("Read Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def plot_read_sweep_multiple(data_dict: dict):
    fig, ax = plt.subplots()
    for key in data_dict.keys():
        plot_read_sweep_single(data_dict, key, ax)
    return ax


def plot_sweep_waterfall(data_dict: dict):
    fig, ax = plt.subplot_mosaic(
        [["waterfall"]], figsize=(16, 9), subplot_kw={"projection": "3d"}
    )
    ax["waterfall"] = plot_waterfall(data_dict, ax=ax["waterfall"])
    write_current = data_dict[0]["write_current"].flatten()[0] * 1e6
    # ax["waterfall"].set_title(f"$I_{{EW}}$ = {enable_write_current:.1f}$\mu$A")
    # ax["waterfall"].text2D(
    #     0.25,
    #     0.6,
    #     f"$I_{{W}}$ = {write_current:.1f}$\mu$A",
    #     transform=ax["waterfall"].transAxes,
    # )
    fig.tight_layout()
    plt.savefig(f"write_current_read_sweep{int(write_current)}.pdf")
    plt.show()


def plot_analytical(data_dict: dict, persistent_current=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    color_map = plt.get_cmap("viridis")
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


if __name__ == "__main__":
    current_cell = "C4"
    HTRON_SLOPE = CELLS[current_cell]["slope"]
    HTRON_INTERCEPT = CELLS[current_cell]["intercept"]
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.213
    ALPHA = 0.563

    MAX_CRITICAL_CURRENT = 860e-6  # CELLS[current_cell]["max_critical_current"]
    IRETRAP_ENABLE = 0.573
    IREAD = 630
    N = 200

    write_read_sweep_C4_dict = {
        0: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 17-48-03.mat"
        ),
        1: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 17-39-50.mat"
        ),
        2: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 16-28-13.mat"
        ),
        3: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 16-09-02.mat"
        ),
        4: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 16-17-33.mat"
        ),
        5: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 17-56-41.mat"
        ),
        6: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 18-11-47.mat"
        ),
        7: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 18-19-41.mat"
        ),
        8: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 18-28-51.mat"
        ),
        9: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 18-36-42.mat"
        ),
        10: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 19-18-14.mat"
        ),
        11: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 19-26-34.mat"
        ),
        12: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 19-34-47.mat"
        ),
    }

    write_read_sweep_C4_dict_min_pulsewidth = {
        0: load_data(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-51-16.mat"
        ),
        1: load_data(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-47-12.mat"
        ),
        2: load_data(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-43-05.mat"
        ),
        3: load_data(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-36-19.mat"
        ),
        4: load_data(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-31-59.mat"
        ),
        5: load_data(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-27-18.mat"
        ),
        6: load_data(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-05-17.mat"
        ),
        7: load_data(
            "SPG806_20241023_nMem_parameter_sweep_D6_A4_C4_2024-10-23 09-00-10.mat"
        ),
    }
    plot_read_sweep_multiple(write_read_sweep_C4_dict_min_pulsewidth)
