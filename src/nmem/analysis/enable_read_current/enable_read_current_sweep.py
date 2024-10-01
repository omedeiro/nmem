import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D

from nmem.measurement.cells import CELLS

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 14


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.0), *zip(x, y), (x[-1], 0.0)]


def plot_waterfall(data_dict: dict, ax: Axes3D = None):
    if ax is None:
        fig, ax = plt.subplots(projection="3d")
    plt.sca(ax)
    cmap = plt.get_cmap("viridis")
    # cmap = cmap.reversed()
    colors = cmap(np.linspace(0, 1, len(data_dict)))
    verts_list = []
    zlist = []
    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        current_cell = data["cell"][0]
        ber = data["bit_error_rate"].flatten()
        enable_read_current = data["enable_read_current"].flatten()[0] * 1e6
        zlist.append(enable_read_current)
        verts = polygon_under_graph(read_currents, ber)
        verts_list.append(verts)
    poly = PolyCollection(verts_list, facecolors=colors, alpha=0.6, edgecolors="k")
    ax.add_collection3d(poly, zs=zlist, zdir="y")

    # ax.set_xlabel("$I_{{EW}}$ ($\mu$A)", labelpad=10)
    ax.set_xlabel("$I_R$ ($\mu$A)", labelpad=10)
    ax.set_ylabel("$I_{{ER}}$ ($\mu$A)", labelpad=70)
    ax.set_zlabel("BER", labelpad=10)
    # ax.set_zscale("log")
    ax.tick_params(axis="both", which="major", labelsize=12, pad=5)

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)

    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlim(590, 950)
    ax.set_ylim(zlist[0], zlist[-1])
    # ax.set_xticks(np.linspace(300, 600, 4))
    # ax.set_yticks(zlist)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect([0.5, 1, 0.2], zoom=0.8)
    ax.view_init(20, -35)
    ax.grid(False)
    ax.grid(axis="x", linestyle=":", color="lightgray")
    ax = plt.gca()
    return ax


def load_data(file_path: str):
    data = sio.loadmat(file_path)
    return data


def find_edge(data: np.ndarray) -> list:
    pos_data = np.argwhere(data > 0.6)
    neg_data = np.argwhere(data < 0.4)

    if len(pos_data)>0:
        pos_edge1 = pos_data[0][0]
        neg_edge1 = pos_data[-1][0]
    else:
        pos_edge1 = 0
        neg_edge1 = 0
    if len(neg_data)>0:
        neg_edge2 = neg_data[0][0]
        pos_edge2 = neg_data[-1][0]
    else:
        neg_edge2 = 0
        pos_edge2 = 0
    return [pos_edge1, neg_edge1, neg_edge2, pos_edge2]

def get_edges(data_dict: dict):
    edges = []
    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        edges.append(find_edge(ber))

    return edges


def plot_read_sweep_single(data_dict: dict, index: int, ax=None, fill=False):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.3, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        read_currents = data["y"][:, :, 0].flatten() * 1e6
        ber = data["bit_error_rate"].flatten()
        enable_read = data["enable_read_current"].flatten()[0] * 1e6
        plt.plot(
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
            plt.vlines(
                read_currents[edge],
                0,
                1,
                linestyle=":",
                color=colors[key],
            )
        print(f"read current edges: {read_currents[edges]}")

    if fill:
        plt.fill_between(
            read_currents,
            ber,
            np.ones_like(ber) * 0.5,
            color=colors[key],
            alpha=0.5,
        )

    ax = plt.gca()
    plt.xlabel("Read Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")
    plt.yticks(np.linspace(0, 1, 5))
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def plot_write_sweep_single(data_dict: dict, index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.8, len(data_dict)))

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
    plt.savefig("enable_write_current_sweep.pdf", format="pdf", bbox_inches="tight")
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

    enable_read_long_dict = {
        0: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-10-30.mat"
        ),
        1: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-19-12.mat"
        ),
        2: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-26-55.mat"
        ),
        3: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-33-48.mat"
        ),
        4: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-40-47.mat"
        ),
        5: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-49-39.mat"
        ),
        6: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 12-59-27.mat"
        ),
        7: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-10-02.mat"
        ),
        8: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-17-53.mat"
        ),
        9: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-24-46.mat"
        ),
        10: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-32-29.mat"
        ),
        11: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-40-00.mat"
        ),
        12: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 13-53-35.mat"
        ),
        13: load_data(
            "SPG806_20240930_nMem_parameter_sweep_D6_A4_C1_2024-09-30 14-02-47.mat"
        ),
    }


    enable_read_short_dict = {
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
    # plot_enable_read_sweep_multiple(data_dict)
    # plot_enable_write_sweep_multiple(write_dict)

    plot_enable_read_sweep_multiple(enable_read_long_dict)

    # plot_sweep_waterfall(enable_read_long_dict)
