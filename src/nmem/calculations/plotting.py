import matplotlib.pyplot as plt
import numpy as np
from nmem.calculations.calculations import (
    calculate_persistent_current,
    calculate_read_currents,
)


def plot_point(ax, x, y, **kwargs):
    ax.plot(x, y, **kwargs)
    return ax


def plot_htron_critical_current(
    enable_currents: np.ndarray, critical_currents: np.ndarray
):
    plt.plot(enable_currents, critical_currents)
    plt.xlabel("Enable Current [uA]")
    plt.ylabel("Critical Current [uA]")
    plt.title("Critical Current vs Enable Current")
    plt.show()


def print_dict_keys(data_dict: dict):
    for key in data_dict.keys():
        print(key)


def plot_htron_sweep(
    ax: plt.Axes,
    write_currents: np.ndarray,
    enable_write_currents: np.ndarray,
    ber: np.ndarray,
):
    xx, yy = np.meshgrid(enable_write_currents, write_currents)
    plt.contourf(xx, yy, ber)
    # plt.gca().invert_xaxis()
    plt.xlabel("Enable Current [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("BER vs Write Current and Critical Current")
    plt.colorbar()
    return ax


def plot_edge_region(c, mask: np.ndarray, color="red", edge_color_array=None):
    edge_color_list = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                edge_color_list.append(color)
            else:
                if edge_color_array is not None:
                    edge_color_list.append(edge_color_array[i, j])
                else:
                    edge_color_list.append("none")
    c.set_edgecolor(edge_color_list)
    return c, edge_color_list


def plot_mask_region(
    c,
    mask_list: list,
    mask_names: list,
    colors=["r", "g", "b", "y"],
    edge_color_array=None,
):
    edge_color_array = None
    for i, mask in enumerate(mask_list):
        c, edge_color_list = plot_edge_region(
            c, mask, color=colors[i], edge_color_array=edge_color_array
        )
        edge_color_array = np.array(edge_color_list).reshape(mask.shape)

    return c


def plot_persistent_current(
    ax,
    data_dict: dict,
    plot_regions=False,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    width_ratio = data_dict["width_ratio"]

    total_persistent_current, regions = calculate_persistent_current(data_dict)

    c = plt.pcolormesh(
        left_critical_currents_mesh,
        write_currents_mesh,
        total_persistent_current,
        edgecolors="none",
        linewidth=0.5,
    )

    if plot_regions:
        mask_list = [
            regions["left_switch"],
            regions["right_switch"],
            regions["right_retrap"],
            regions["left_persistent_switch"],
        ]
        mask_names = [
            "left_switch",
            "right_switch",
            "right_retrap",
            "left_persistent_switch",
        ]
        c = plot_mask_region(c, mask_list, mask_names)

    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("Maximum Persistent Current")
    plt.gca().invert_xaxis()
    plt.colorbar()
    ax.set_xlim(right=0)

    # plt.text(
    #     60,
    #     80,
    #     "Write too low",
    #     fontsize=12,
    #     color="red",
    #     ha="left",
    #     backgroundcolor="white",
    # )
    # plt.text(
    #     40,
    #     220,
    #     "Switched right side, inverting",
    #     fontsize=12,
    #     color="green",
    #     ha="left",
    #     backgroundcolor="white",
    # )
    # plt.text(
    #     10,
    #     170,
    #     "I_P > ICHL",
    #     fontsize=12,
    #     color="blue",
    #     ha="center",
    #     backgroundcolor="white",
    # )

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels(
        [f"{ic*(1-width_ratio)/width_ratio:.0f}" for ic in ax.get_xticks()]
    )
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")
    return ax, total_persistent_current, regions


def plot_read_current(
    ax: plt.Axes,
    data_dict: dict,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    persistent_currents = data_dict["persistent_currents"]
    width_ratio = data_dict["width_ratio"]
    set_read_current = data_dict["set_read_current"]

    read_currents, read_margins = calculate_read_currents(data_dict)

    read_currents = np.where(read_currents < write_currents_mesh, 0, read_currents)
    read_currents = np.where(persistent_currents == 0, 0, read_currents)

    plt.pcolormesh(
        left_critical_currents_mesh, write_currents_mesh, read_currents, linewidth=0.5
    )
    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("Read Current")
    plt.gca().invert_xaxis()
    plt.colorbar()
    ax.set_xlim(right=0)

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels(
        [f"{ic*(1-width_ratio)/width_ratio:.0f}" for ic in ax.get_xticks()]
    )
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")
    return ax, read_currents, read_margins


def plot_edge_fits(ax, lines, critical_currents):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for line in lines:
        plot_edge_fit(ax, critical_currents, **line)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return ax


def plot_edge_fit(ax, x, p1, p2):
    y = p1 * x + p2
    ax.plot(x, y, color="red")
    return ax
