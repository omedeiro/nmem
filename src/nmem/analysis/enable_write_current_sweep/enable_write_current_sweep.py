import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from nmem.measurement.cells import CELLS

plt.rcParams["figure.figsize"] = [6, 4]
plt.rcParams["font.size"] = 12


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

    # ax.set_xlabel("$I_{{EW}}$ ($\mu$A)", labelpad=10)
    ax.set_xlabel("$I_{{C, H_L}}$ ($\mu$A)", labelpad=10)
    ax.set_ylabel("$I_W$ ($\mu$A)", labelpad=70)
    ax.set_zlabel("BER", labelpad=10)
    # ax.set_zscale("log")
    ax.tick_params(axis="both", which="major", labelsize=12, pad=5)

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)

    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlim(250, 340)
    ax.set_ylim(10, zlist[-1])
    ax.set_xticks(np.linspace(300, 600, 4))
    ax.set_yticks(zlist)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect([0.5, 1, 0.2], zoom=0.8)
    ax.view_init(20, -35)
    ax.grid(False)
    # ax.grid(axis="x", linestyle=":", color="lightgray")
    return ax


def find_operating_peaks(data_dict: dict):
    bit_error_rate = data_dict["bit_error_rate"].flatten()
    enable_write_currents = data_dict["x"][:, :, 0].flatten() * 1e6

    nominal_peak = np.mean(enable_write_currents[bit_error_rate < 0.3])
    inverting_peak = np.mean(enable_write_currents[bit_error_rate > 0.7])

    return nominal_peak, inverting_peak


def find_operating_width(data_dict: dict):
    bit_error_rate = data_dict["bit_error_rate"].flatten()
    enable_write_currents = data_dict["x"][:, :, 0].flatten() * 1e6

    nominal_peak = enable_write_currents[bit_error_rate < 0.4]
    nominal_width = nominal_peak[-1] - nominal_peak[0]
    inverting_peak = enable_write_currents[bit_error_rate > 0.6]
    inverting_width = inverting_peak[-1] - inverting_peak[0]

    return nominal_width, inverting_width


def get_operating_widths(data_dict: dict):
    widths = []
    write_currents = []
    for key in data_dict.keys():
        write_currents.append(data_dict[key]["write_current"].flatten()[0] * 1e6)
        nominal_width, inverting_width = find_operating_width(data_dict[key])
        widths.append((nominal_width, inverting_width))
    return write_currents, widths


def get_operating_peaks(data_dict: dict):
    peaks = []
    write_currents = []
    for key in data_dict.keys():
        write_currents.append(data_dict[key]["write_current"].flatten()[0] * 1e6)
        nominal_peak, inverting_peak = find_operating_peaks(data_dict[key])
        peaks.append((nominal_peak, inverting_peak))
    return write_currents, peaks


def plot_enable_write_sweep_single(
    data_dict: dict, index: int, ax=None, find_peaks=False
):
    if ax is None:
        fig, ax = plt.subplots()
    plt.sca(ax)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.3, 1, len(data_dict)))

    data_dict = {index: data_dict[index]}

    for key, data in data_dict.items():
        enable_write_currents = data["x"][:, :, 0].flatten() * 1e6
        current_cell = data["cell"][0]
        left_critical_currents = (
            enable_write_currents * CELLS[current_cell]["slope"]
        ) + CELLS[current_cell]["intercept"]

        ber = data["bit_error_rate"].flatten()
        write_current = data["write_current"].flatten()[0] * 1e6
        plt.plot(
            left_critical_currents,
            ber,
            label=f"$I_{{W}}$ = {write_current:.1f}$\mu$A",
            color=colors[key],
            marker=".",
            markeredgecolor="k",
        )
    width_ratio = 3.0
    ax = plt.gca()

    # plt.xlim(570, 650)
    # plt.hlines([0.5], ax.get_xlim()[0], ax.get_xlim()[1], linestyle=":", color="lightgray")
    # plt.ylim(1e-4, 1)
    # plt.xticks(np.linspace(570, 650, 5))
    # plt.yscale("log")

    plt.xlabel("Left Critical Current ($\mu$A)")
    plt.ylabel("Bit Error Rate")
    plt.grid(True)
    plt.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    # plt.title("write_1_read_0_norm")

    if find_peaks:
        nominal_peak, inverting_peak = find_operating_peaks(data_dict[index])
        plt.axvline(nominal_peak, color="r", linestyle="--")
        plt.axvline(inverting_peak, color="r", linestyle="--")
        # print(f"Nominal peak: {nominal_peak:.1f} uA")
        # print(f"Inverting peak: {inverting_peak:.1f} uA")
        print(
            f"Write current: {write_current:.1f} uA, Peak distance: {inverting_peak - nominal_peak:.1f} uA"
        )
    return ax


def plot_enable_write_sweep_multiple(data_dict: dict, find_peaks=False):
    fig, ax = plt.subplots()
    for key in data_dict.keys():
        plot_enable_write_sweep_single(data_dict, key, ax, find_peaks=find_peaks)

    width_ratio = 3.0
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ichl*width_ratio:.0f}" for ichl in ax.get_xticks()])

    plt.show()


def plot_peak_distance(write_currents, peaks):
    fig, ax = plt.subplots()
    plt.plot(
        write_currents,
        [x[1] - x[0] for x in peaks],
        label="Peak Distance",
        marker="o",
        color="C4",
    )
    # plot a linear fit
    fit_x = np.array(write_currents[5:-1])
    fit_y = np.array([x[1] - x[0] for x in peaks[5:-1]])
    m, b = np.polyfit(fit_x, fit_y, 1)
    plt.plot(
        fit_x,
        fit_y,
        label="Peak Distance",
        marker="o",
        linestyle="None",
        markeredgecolor="k",
        markerfacecolor="None",
    )
    plt.plot(fit_x, m * fit_x + b, label=f"y = {m:.2f}x + {b:.2f}", color="k")
    plt.text(20, 40, f"y = {m:.2f}x + {b:.2f}", fontsize=12)
    plt.xlabel("Write Current ($\mu$A)")
    plt.ylabel("Peak Distance ($\mu$A)")
    plt.grid(True)
    plt.show()


def plot_peak_widths(write_currents, peaks):
    fig, ax = plt.subplots()
    write_currents, widths = get_operating_widths(data_dict)
    plt.plot(write_currents, [x[0] for x in widths], label="Nominal Width")
    plt.plot(write_currents, [x[1] for x in widths], label="Inverting Width")
    plt.xlabel("Write Current ($\mu$A)")
    plt.ylabel("Width ($\mu$A)")
    plt.legend()
    plt.show()


def plot_peak_locations(write_currents, peaks):
    fig, ax = plt.subplots()
    plt.plot(write_currents, [x[0] for x in peaks], label="Nominal Peak")
    plt.plot(write_currents, [x[1] for x in peaks], label="Inverting Peak")
    plt.legend()
    plt.show()


def load_data(file_path: str):
    data = sio.loadmat(file_path)
    return data


if __name__ == "__main__":
    data_dict = {
        0: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-52-33.mat"
        ),
        1: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-45-20.mat"
        ),
        2: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-33-31.mat"
        ),
        3: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-26-47.mat"
        ),
        4: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 17-20-06.mat"
        ),
        5: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-22-38.mat"
        ),
        6: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-41-06.mat"
        ),
        7: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-09-12.mat"
        ),
        8: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-14-52.mat"
        ),
        9: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-49-37.mat"
        ),
        10: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-41-43.mat"
        ),
        11: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 19-49-23.mat"
        ),
        12: load_data(
            "SPG806_20240924_nMem_parameter_sweep_D6_A4_C1_2024-09-24 18-19-05.mat"
        ),
    }

    data_dict1 = {
        0: data_dict[5],
    }

    plot_enable_write_sweep_multiple(data_dict, find_peaks=False)

    write_currents, peaks = get_operating_peaks(data_dict)
    # plot_peak_distance(write_currents, peaks)
    # plot_peak_locations(write_currents, peaks)
    # plot_peak_widths(write_currents, peaks)

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
    fig, ax = plt.subplot_mosaic(
        [["waterfall"]], figsize=(16, 9), subplot_kw={"projection": "3d"}
    )
    ax["waterfall"] = plot_waterfall(data_dict2, ax=ax["waterfall"])
    plt.savefig("enable_write_current_sweep.pdf", format="pdf", bbox_inches="tight")
    plt.show()
