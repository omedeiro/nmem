import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import patches
from scipy.signal import find_peaks

from nmem.calculations.analytical_model import create_dict_read

from nmem.measurement.cells import CELLS
from nmem.analysis.analysis import (
    load_data,
    find_edge,
    polygon_nominal,
    polygon_inverting,
    plot_analytical,
)
from nmem.analysis.enable_read_current.trace_plotting import (
    text_from_bit,
    plot_threshold,
    plot_data_delay_manu,
    INVERSE_COMPARE_DICT,
)

plt.rcParams["figure.figsize"] = [3.5, 3.5]
plt.rcParams["font.size"] = 6
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
plt.rcParams["lines.markeredgewidth"] = 0.5

plt.rcParams["xtick.major.size"] = 1
plt.rcParams["ytick.major.size"] = 1

CURRENT_CELL = "C1"


def plot_waterfall(data_dict: dict, ax: Axes3D = None):
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "3d"}, layout="none", figsize=(3.5, 3.5)
        )

    cmap = plt.get_cmap("RdBu")
    # cmap = cmap.reversed()
    colors = cmap(np.linspace(0, 1, len(data_dict)))
    verts_list = []
    inv_verts_list = []
    zlist = []
    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        current_cell = data["cell"][0]
        ber = data["bit_error_rate"].flatten()
        enable_read_current = data["enable_read_current"].flatten()[0] * 1e6
        zlist.append(enable_read_current)
        verts = polygon_nominal(read_currents, ber)
        inv_verts = polygon_inverting(read_currents, ber)
        verts_list.append(verts)
        inv_verts_list.append(inv_verts)

    poly = PolyCollection(verts_list, facecolors=colors[-1], alpha=0.6, edgecolors=None)
    poly_inv = PolyCollection(
        inv_verts_list, facecolors=colors[0], alpha=0.6, edgecolors=None
    )
    ax.add_collection3d(poly, zs=zlist, zdir="y")
    ax.add_collection3d(poly_inv, zs=zlist, zdir="y")

    # ax.set_xlabel("$I_{{EW}}$ ($\mu$A)", labelpad=10)
    ax.set_xlabel("$I_R$ ($\mu$A)", labelpad=-6)
    ax.set_ylabel("$I_{{ER}}$ ($\mu$A)", labelpad=4)
    ax.set_zlabel("BER", labelpad=-6)
    # ax.set_zscale("log")
    ax.tick_params(axis="both", which="major", pad=-1)

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)

    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlim(500, 950)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_ylim(zlist[0], zlist[-1])
    ax.set_yticks(zlist)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect([0.5, 1, 0.25], zoom=0.8)
    ax.view_init(20, -35)
    ax.grid(False)

    for key, data in data_dict.items():
        cmap = plt.get_cmap("RdBu")
        colors = cmap(np.linspace(0, 1, 4))
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

    ax.set_position([0.0, 0.0, 1, 1])

    for child in ax.get_children():
        if isinstance(child, Rectangle):
            child.set_visible(False)

    # fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0.1)
    # fig.bbox = Bbox.from_bounds(-x0/4, -100, width*8, height*4)
    # plt.show()
    return ax


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
    cmap = plt.get_cmap("RdBu")
    colors = cmap(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
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
                param, read_current[e], color=colors[i], marker=markers[i], edgecolor=colors[i], linewidth=0.5, s=27
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
            ax.plot(fitx, fity, "o", color="k", fillstyle="none", mew=0.5)
            ax.text(
                1.50,
                0.5 + 0.1 * i,
                f"y = {fit_fn[1]:.3f}x + {fit_fn[0]:.1f}",
                transform=ax.transAxes,
                color=colors[i],
            )

    return ax


def plot_edge_3D(edge_dict: dict, edge_list: list, key: int, colors: list, ax: Axes3D):
    markers = ["o", "s", "D", "^"]
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
            marker=markers[edge_list.index(edge)],
            markeredgecolor=None,
            markeredgewidth=0.5,
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
    # ax.set_aspect("equal")
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
    ax.yaxis.set_major_locator(MultipleLocator(100))

    ax, read_current_dict = plot_analytical(
        analytical_data_dict, persistent_current=persistent_current, ax=ax
    )

    # ax.set_ylim(500, 950)
    # ax.set_xlim(600, 950)

    # plt.title("Enable Read Current Edges")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(direction="in", top=True, right=True, bottom=True, left=True, length=2)
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
    axs=None,
):
    if axs is None:
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

    # fig.supxlabel("$I_{{CH}}$ ($\mu$A)", x=0.5, y=0.04)
    # fig.supylabel("$I_{{R}}$ ($\mu$A)", x=0.99, y=0.5)

    # caxis = fig.add_axes([0.21, 0.9, 0.61, 0.02])
    # cbar = fig.colorbar(
    #     axs[0].collections[-1], cax=caxis, orientation="horizontal", pad=0.1
    # )
    # caxis.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True)
    # plt.savefig("enable_read_current_edges_stack.pdf", bbox_inches="tight")
    # plt.show()
    return axs


def plot_read_sweep_single(data_dict: dict, index: int, ax=None, fill=False):
    if ax is None:
        fig, ax = plt.subplots()
    cmap = plt.get_cmap("RdBu")
    colors = cmap(np.linspace(0, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        enable_read = data["enable_read_current"].flatten()[0] * 1e6
        ax.plot(
            read_currents,
            ber,
            label=f"$I_{{ER}}$ = {enable_read:.1f}$\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )

        edges = find_edge(ber)
        for edge in edges:
            if edge == 0:
                continue
            ax.vlines(
                read_currents[edge],
                0,
                1,
                linestyle=":",
                color=colors[key],
            )
        print(f"read current edges: {read_currents[edges]}")

    if fill:
        ax.fill_between(
            read_currents,
            ber,
            np.ones_like(ber) * 0.5,
            color=colors[key],
            alpha=0.5,
        )
    ax.set_xlabel("Read Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def plot_write_sweep_single(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("RdBu")
    colors = cmap(np.linspace(0, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        enable_write_current = data["enable_write_current"].flatten()[0] * 1e6
        plt.plot(
            read_currents,
            ber,
            label=f"$I_{{EW}}$ = {enable_write_current:.1f}$\mu$A",
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


def plot_write_sweep_single_log(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("RdBu")
    colors = cmap(np.linspace(0, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        if any(ber > 0.9):
            ber = 1 - ber
            line_style = "--"
        else:
            line_style = "-"
        enable_write_current = data["enable_write_current"].flatten()[0] * 1e6
        plt.plot(
            read_currents,
            ber,
            label=f"$I_{{EW}}$ = {enable_write_current:.1f}$\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
            linestyle=line_style,
        )

    ax = plt.gca()
    plt.yscale("log")
    plt.xlabel("Read Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def plot_data_delay_manu_dev(data_dict_keyd, axs=None):
    cmap = plt.get_cmap("RdBu").reversed()
    colors = cmap(np.linspace(0, 1, 8))
    data_dict = data_dict_keyd[0]
    INDEX = 14
    if axs is None:
        fig, axs = plt.subplots(6, 1, figsize=(2.6, 3.54))

    ax = axs[0]
    x = data_dict["trace_chan_in"][0][:, INDEX] * 1e6
    yin = np.mean(data_dict["trace_chan_in"][1], axis=1) * 1e3
    (p1,) = ax.plot(x, yin, color=colors[0], label="Input")
    ax.set_xlim([0, 10])
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(data_dict["bitmsg_channel"][0]):
        text = text_from_bit(bit)
        ax.text(
            i + 0.6,
            axheight * 1.1,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0,
        )
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim([-150, 1100])
    ax.set_yticks([0, 500, 1200])
    ax.tick_params(
        direction="in", top=True, bottom=True, right=True, left=True, length=2
    )

    ax = axs[1]
    x = data_dict["trace_enab"][0][:, INDEX] * 1e6
    y = np.mean(data_dict["trace_enab"][1], axis=1) * 1e3
    (p2,) = ax.plot(x, y, color=colors[-1], label="Enable")
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(data_dict["bitmsg_enable"][0]):
        text = text_from_bit(bit)
        ax.text(
            i + 0.5,
            axheight * 0.96,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0,
        )
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim([-10, 100])
    ax.set_xlim([0, 10])
    ax.set_yticks([0, 50])
    ax.tick_params(
        direction="in", top=True, bottom=True, right=True, left=True, length=2
    )

    ax = axs[2]
    x = data_dict["trace_chan_out"][0][:, INDEX] * 1e6
    yout = data_dict["trace_chan_out"][1][:, INDEX] * 1e3
    # yout = np.roll(yout, 10)*1.3
    (p3,) = ax.plot(x, yout, color=colors[1], label="Output")
    # plt.grid(axis="x", which="both")
    ax = plot_threshold(ax, 4, 5, 400)
    ax = plot_threshold(ax, 9, 10, 400)
    ax.set_ylim([-150, 900])
    ax.set_xlim([0, 10])
    ax.set_yticks([0, 500])
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate("NNNNzNNNNo"):
        text = text_from_bit(bit)
        ax.text(
            i + 0.5,
            axheight * 0.7,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0,
        )
    ax.tick_params(direction="in")
    ax.set_xticklabels([])
    ax.set_xlim([0, 10])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(
        direction="in", top=True, bottom=True, right=True, left=True, length=2
    )

    ax = axs[3]
    data_dict = data_dict_keyd[1]
    x = data_dict["trace_chan_out"][0][:, INDEX] * 1e6
    yout = data_dict["trace_chan_out"][1][:, INDEX] * 1e3
    (p3,) = ax.plot(x, yout, color=colors[1], label="Output")
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate("NNNNZNNNNO"):
        text = text_from_bit(bit)
        ax.text(
            i + 0.2,
            axheight * 0.95,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0,
        )
    ax = plot_threshold(ax, 4, 5, 400)
    ax = plot_threshold(ax, 9, 10, 400)
    ax.set_ylim([-150, 900])
    ax.set_xlim([0, 10])
    ax.yaxis.set_ticks([0, 500])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(
        direction="in", top=True, bottom=True, right=True, left=True, length=2
    )

    ax = axs[4]
    data_dict = data_dict_keyd[2]
    x = data_dict["trace_chan_out"][0][:, INDEX] * 1e6
    yout = data_dict["trace_chan_out"][1][:, INDEX] * 1e3
    (p3,) = ax.plot(x, yout, color=colors[1], label="Output")
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate("NNNNzNNNNO"):
        text = text_from_bit(bit)
        ax.text(
            i + 0.2,
            axheight * 1.2,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0,
        )
    ax = plot_threshold(ax, 4, 5, 400)
    ax = plot_threshold(ax, 9, 10, 400)
    ax.set_ylim([-150, 900])
    ax.set_xlim([0, 10])
    ax.yaxis.set_ticks([0, 500])
    ax.set_xticklabels([])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(
        direction="in", top=True, bottom=True, right=True, left=True, length=2
    )

    ax = axs[5]
    data_dict = data_dict_keyd[2]
    x = data_dict["trace_chan_out"][0][:, -1] * 1e6
    yout = data_dict["trace_chan_out"][1][:, -1] * 1e3
    (p3,) = ax.plot(x, yout, color=colors[1], label="Output")
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate("NNNNZNNNNo"):
        text = text_from_bit(bit)
        ax.text(
            i + 0.51,
            axheight * 1.0,
            text,
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0,
        )
    ax = plot_threshold(ax, 4, 5, 400)
    ax = plot_threshold(ax, 9, 10, 400)
    ax.set_ylim([-150, 900])
    ax.set_xlim([0, 10])
    ax.set_yticks([0, 500])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(
        direction="in", top=True, bottom=True, right=True, left=True, length=2
    )
    ax.set_xticklabels(["", "0", "", "", "", "", "5", "", "", "", "", "10"])
    return axs


def plot_enable_read_sweep_multiple(data_dict: dict):
    fig, ax = plt.subplots()
    for key in data_dict.keys():
        plot_read_sweep_single(data_dict, key, ax)
    return ax


def plot_enable_write_sweep_multiple(data_dict: dict):
    fig, ax = plt.subplots()
    for key in data_dict.keys():
        plot_write_sweep_single(data_dict, key, ax)
    return ax


def plot_sweep_waterfall(data_dict: dict):
    fig, ax = plt.subplot_mosaic(
        [["waterfall"]], figsize=(16, 9), subplot_kw={"projection": "3d"}
    )
    ax["waterfall"] = plot_waterfall(data_dict, ax=ax["waterfall"])
    enable_write_current = data_dict[0]["enable_write_current"].flatten()[0] * 1e6
    # ax["waterfall"].set_title(f"$I_{{EW}}$ = {enable_write_current:.1f}$\mu$A")
    ax["waterfall"].text2D(
        0.25,
        0.6,
        f"$I_{{EW}}$ = {enable_write_current:.1f}$\mu$A",
        transform=ax["waterfall"].transAxes,
    )
    fig.tight_layout()
    plt.savefig(f"enable_read_current_sweep_{int(enable_write_current)}.pdf")
    plt.show()


def manuscript_figure(data_dict):
    fig = plt.figure(figsize=(9.5, 3.5))

    subfigs = fig.subfigures(1, 3, wspace=-0.3, width_ratios=[0.5, 1, 1])

    axs = subfigs[0].subplots(6, 1, sharex=True, sharey=False)
    axstrace = plot_data_delay_manu_dev(INVERSE_COMPARE_DICT, axs)
    subfigs[0].supxlabel("Time ($\mu$s)", x=0.5, y=-0.01)
    subfigs[0].supylabel("Voltage (mV)", x=1.01, y=0.5, rotation=270)
    subfigs[0].subplots_adjust(hspace=0.0, bottom=0.05, top=0.95)

    axsslice = subfigs[1].subplots(3, 1, subplot_kw={"projection": "3d"})
    axsslice[0] = plot_waterfall(enable_read_290_dict, ax=axsslice[0])
    axsslice[1] = plot_waterfall(enable_read_300_dict, ax=axsslice[1])
    axsslice[2] = plot_waterfall(enable_read_310_dict, ax=axsslice[2])
    subfigs[1].subplots_adjust(
        hspace=-0.6, bottom=-0.2, top=1.20, left=0.1, right=1.1
    )

    axsstack = subfigs[2].subplots(3, 1, sharex=True, sharey=True)
    axsstack = plot_stack(
        [enable_read_290_dict, enable_read_300_dict, enable_read_310_dict],
        [analytical_data_dict, analytical_data_dict, analytical_data_dict],
        [-30, 0, 30],
        [fitting_dict[-30], fitting_dict[0], fitting_dict[30]],
        axs=axsstack,
    )
    subfigs[2].subplots_adjust(hspace=0.0, bottom=0.06, top=0.90, left=-0.2, right=0.9)
    subfigs[2].supxlabel("$I_{{CH}}$ ($\mu$A)", x=0.36, y=-0.02)
    subfigs[2].supylabel("$I_{{R}}$ ($\mu$A)", x=0.48, y=0.5)

    caxis = subfigs[2].add_axes([0.266, 0.91, 0.165, 0.02])
    cbar = subfigs[2].colorbar(
        axsstack[0].collections[-1], cax=caxis, orientation="horizontal", pad=0.1
    )
    caxis.tick_params(labeltop=True, labelbottom=False, bottom=False, top=True, direction="out")
    fig.patch.set_visible(False)
    plt.savefig("trace_waterfall_fit_combined.pdf", bbox_inches="tight")    
    plt.show()


def plot_trace_only():    
    fig, axs = plt.subplots(6, 1, figsize=(6.6, 3.54))
    axs = plot_data_delay_manu_dev(INVERSE_COMPARE_DICT, axs)
    fig.subplots_adjust(hspace=0.0, bottom=0.05, top=0.95)
    fig.supxlabel("Time ($\mu$s)", x=0.5, y=-0.03)
    fig.supylabel("Voltage (mV)", x=0.95, y=0.5, rotation=270)
    plt.savefig("trace_only.png", bbox_inches="tight", dpi=300)
    plt.show()



if __name__ == "__main__":
    data0 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 15-10-41.mat"
    )
    data1 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 15-17-47.mat"
    )
    data2 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 16-11-46.mat"
    )
    data3 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 16-18-32.mat"
    )
    data4 = sio.loadmat(
        "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 16-25-36.mat"
    )

    write_dict = {
        0: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-23-55.mat"
        ),
        1: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-31-23.mat"
        ),
    }

    data_dict = {0: data0, 1: data1, 2: data2, 3: data3, 4: data4}

    enable_read_290_dict_full = {
        0: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-45-11.mat"
        ),
        1: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-53-18.mat"
        ),
        2: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 16-08-51.mat"
        ),
        3: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 16-19-03.mat"
        ),
        4: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-10-30.mat"
        ),
        5: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-19-12.mat"
        ),
        6: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-26-55.mat"
        ),
        7: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-33-48.mat"
        ),
        8: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-40-47.mat"
        ),
        9: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-49-39.mat"
        ),
        10: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-59-27.mat"
        ),
        11: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-10-02.mat"
        ),
        12: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-17-53.mat"
        ),
        13: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-24-46.mat"
        ),
        14: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-32-29.mat"
        ),
        15: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-40-00.mat"
        ),
        16: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-53-35.mat"
        ),
        17: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-02-47.mat"
        ),
    }

    enable_read_290_dict = {
        0: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-45-11.mat"
        ),
        1: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-53-18.mat"
        ),
        2: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 16-08-51.mat"
        ),
        3: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 16-19-03.mat"
        ),
        4: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-19-12.mat"
        ),
        5: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-33-48.mat"
        ),
        6: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-49-39.mat"
        ),
        7: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-10-02.mat"
        ),
        8: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-24-46.mat"
        ),
        9: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-40-00.mat"
        ),
        10: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-02-47.mat"
        ),
    }

    enable_read_290_dict_short = {
        0: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-19-12.mat"
        ),
        1: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-10-02.mat"
        ),
        2: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-24-46.mat"
        ),
    }

    enable_read_300_dict = {
        0: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 14-54-06.mat"
        ),
        1: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-06-21.mat"
        ),
        2: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 14-44-04.mat"
        ),
        3: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-16-22.mat"
        ),
        4: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-23-31.mat"
        ),
        5: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-30-28.mat"
        ),
        6: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-39-15.mat"
        ),
        7: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-47-05.mat"
        ),
        8: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 15-54-14.mat"
        ),
        9: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 16-04-36.mat"
        ),
        10: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 14-29-17.mat"
        ),
    }

    enable_read_310_dict = {
        0: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-37-34.mat"
        ),
        1: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-30-17.mat"
        ),
        2: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-21-01.mat"
        ),
        3: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-13-38.mat"
        ),
        4: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-15-11.mat"
        ),
        5: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-27-11.mat"
        ),
        6: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-34-04.mat"
        ),
        7: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-44-33.mat"
        ),
        8: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-56-20.mat"
        ),
        9: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 15-05-30.mat"
        ),
        10: load_data(
            "SPG806_20241001_nMem_parameter_sweep_D6_A4_C1_2024-10-01 09-17-27.mat"
        ),
    }

    inverse_compare_dict = {
        0: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-31-23.mat"
        ),
        1: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 09-23-55.mat"
        ),
    }

    # plot_enable_read_sweep_multiple(data_dict)
    # plot_enable_write_sweep_multiple(write_dict)

    # plot_enable_write_sweep_multiple(inverse_compare_dict)

    # plot_sweep_waterfall(enable_read_long_dict)

    current_cell = "C1"
    HTRON_SLOPE = CELLS[current_cell]["slope"]
    HTRON_INTERCEPT = CELLS[current_cell]["intercept"]
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.213
    ALPHA = 0.563

    MAX_CRITICAL_CURRENT = 860e-6  # CELLS[current_cell]["max_critical_current"]
    IRETRAP_ENABLE = 0.573
    IREAD = 630
    N = 200

    enable_read_currents = np.linspace(0, 400, N)
    read_currents = np.linspace(400, 1050, N)

    analytical_data_dict = create_dict_read(
        enable_read_currents,
        read_currents,
        WIDTH_LEFT,
        WIDTH_RIGHT,
        ALPHA,
        IRETRAP_ENABLE,
        MAX_CRITICAL_CURRENT,
        HTRON_SLOPE,
        HTRON_INTERCEPT,
    )

    fitting_dict = {
        -30: {
            0: {"fit_start": 0, "fit_stop": 0},
            1: {"fit_start": 0, "fit_stop": 0},
            2: {"fit_start": 0, "fit_stop": 0},
            3: {"fit_start": 0, "fit_stop": 0},
        },
        0: {
            0: {"fit_start": 1, "fit_stop": 0},
            1: {"fit_start": 0, "fit_stop": 2},
            2: {"fit_start": 0, "fit_stop": 1},
            3: {"fit_start": 0, "fit_stop": 2},
        },
        30: {
            0: {"fit_start": 1, "fit_stop": 0},
            1: {"fit_start": 0, "fit_stop": 2},
            2: {"fit_start": 0, "fit_stop": 5},
            3: {"fit_start": 0, "fit_stop": 1},
        },
    }
    # persistent_current_plot(data_dict)
    # analytical_data_dict = plot_analytical(analytical_data_dict, persistent_current=30)

    # plot_enable_read_current_edges(
    #     enable_read_290_dict, analytical_data_dict, -30, fitting_dict=fitting_dict[-30]
    # )
    # plot_enable_read_current_edges(
    #     enable_read_300_dict, analytical_data_dict, 0, fitting_dict=fitting_dict[0]
    # )
    # plot_enable_read_current_edges(
    #     enable_read_310_dict,
    #     analytical_data_dict,
    #     persistent_current=30,
    #     fitting_dict=fitting_dict[30],
    # )


    # plot_stack(
    #     [enable_read_290_dict, enable_read_300_dict, enable_read_310_dict],
    #     [analytical_data_dict, analytical_data_dict, analytical_data_dict],
    #     [-30, 0, 30],
    #     [fitting_dict[-30], fitting_dict[0], fitting_dict[30]],
    # )
    # plot_sweep_waterfall(enable_read_310_dict)
    enable_read_310_C4_dict = {
        0: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 13-49-54.mat"
        ),
        1: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-03-11.mat"
        ),
        2: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-10-42.mat"
        ),
        3: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-17-31.mat"
        ),
        4: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-24-30.mat"
        ),
        5: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 15-43-26.mat"
        ),
        6: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-45-10.mat"
        ),
        7: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 14-52-08.mat"
        ),
        8: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 15-06-17.mat"
        ),
        9: load_data(
            "SPG806_20241016_nMem_parameter_sweep_D6_A4_C4_2024-10-16 15-13-23.mat"
        ),
    }

    # manuscript_figure(data_dict)
    fig, axs = plt.subplots(1, 1, figsize=(2.6, 3.54))
    axs = plot_enable_read_current_edges_stack(enable_read_310_dict, analytical_data_dict, axs, 30, fitting_dict[30])
    axs.set_xlabel("$I_{CH}$ ($\mu$A)")
    axs.set_ylabel("$I_{R}$ ($\mu$A)")
    fig.tight_layout()
    plt.savefig("slide_analytical_fit.png", bbox_inches="tight", dpi=300)
    plt.show()

    plot_trace_only()

    plot_waterfall(enable_read_310_dict)
    fig = plt.gcf()
    fig.patch.set_visible(False)
    plt.savefig("slide_waterfall.png", bbox_inches="tight", dpi=300)