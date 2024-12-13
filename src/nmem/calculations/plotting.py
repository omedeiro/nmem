import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes import Axes


def plot_htron_critical_current(
    enable_currents: np.ndarray, critical_currents: np.ndarray
) -> None:
    plt.plot(enable_currents, critical_currents)
    plt.xlabel("Enable Current [uA]")
    plt.ylabel("Critical Current [uA]")
    plt.title("Critical Current vs Enable Current")
    plt.show()
    return


def print_dict_keys(data_dict: dict) -> None:
    for key in data_dict.keys():
        print(key)
    return


def plot_htron_sweep(
    ax: Axes,
    write_currents: np.ndarray,
    enable_write_currents: np.ndarray,
    ber: np.ndarray,
) -> Axes:
    xx, yy = np.meshgrid(enable_write_currents, write_currents)
    ax.pcolormesh(xx, yy, ber, vmin=0, vmax=1)

    ax.set_xlabel("Enable Current [uA]")
    ax.set_ylabel("Write Current [uA]")
    ax.title("BER vs Write Current and Critical Current")
    cbar = plt.colorbar()
    cbar.set_ticks([0, 0.5, 1])

    return ax


def plot_htron_sweep_scaled(
    ax: Axes,
    left_critical_currents: np.ndarray,
    write_currents: np.ndarray,
    ber: np.ndarray,
) -> Axes:
    xx, yy = np.meshgrid(left_critical_currents, write_currents)
    ax.pcolor(xx, yy, ber, shading="auto", vmin=0, vmax=1)
    ax.set_xlabel("Left Critical Current [uA]")
    ax.set_ylabel("Write Current [uA]")
    ax.title("BER vs Write Current and Critical Current")
    cbar = ax.colorbar()
    cbar.set_ticks([0, 0.5, 1])

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
    colors=["r", "g", "b", "k"],
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
    ax: Axes,
    left_critical_currents_mesh: np.ndarray,
    write_currents_mesh: np.ndarray,
    total_persistent_current: np.ndarray,
    width_ratio: float,
) -> Axes:

    c = ax.pcolormesh(
        left_critical_currents_mesh,
        write_currents_mesh,
        total_persistent_current,
        edgecolors="none",
        linewidth=0.5,
    )

    ax.set_xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    ax.set_ylabel("Write Current [uA]")
    ax.invert_xaxis()

    cbar = plt.colorbar(c)
    cbar.set_label("Maximum Persistent Current [uA]")

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ichl*width_ratio:.0f}" for ichl in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    return ax


def plot_zero_state_currents(
    ax: plt.Axes,
    data_dict: dict,
    plot_regions: bool = False,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    width_ratio = data_dict["width_ratio"]

    zero_state_currents = data_dict["zero_state_currents"]

    c = plt.pcolormesh(
        left_critical_currents_mesh,
        write_currents_mesh,
        zero_state_currents,
        edgecolors="none",
        linewidth=0.5,
        cmap="cividis",
    )

    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Read Current [uA]")
    plt.title("Zero State Current")
    plt.gca().invert_xaxis()
    cbar = plt.colorbar()
    cbar.set_ticks([cbar.vmin, cbar.vmax])

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ichl*width_ratio:.0f}" for ichl in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    if plot_regions:
        ax = plot_region_dict(
            ax, data_dict["regions"], left_critical_currents_mesh, write_currents_mesh
        )

    return ax


def plot_one_state_currents(
    ax: plt.Axes,
    data_dict: dict,
    plot_regions=False,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    width_ratio = data_dict["width_ratio"]

    one_state_currents = data_dict["one_state_currents"]

    c = ax.pcolormesh(
        left_critical_currents_mesh,
        write_currents_mesh,
        one_state_currents,
        edgecolors="none",
        linewidth=0.5,
        cmap="cividis",
    )

    ax.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    ax.ylabel("Read Current [uA]")
    ax.title("One State Current")
    ax.invert_xaxis()
    cbar = plt.colorbar()
    cbar.set_ticks([cbar.vmin, cbar.vmax])

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ichl*width_ratio:.0f}" for ichl in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    if plot_regions:
        ax = plot_region_dict(
            ax, data_dict["regions"], left_critical_currents_mesh, write_currents_mesh
        )

    return ax


def plot_state_currents(data_dict: dict, plot_regions=False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.sca(axes[0])
    axes[0] = plot_zero_state_currents(axes[0], data_dict, plot_regions=plot_regions)
    plt.sca(axes[1])
    axes[1] = plot_one_state_currents(axes[1], data_dict, plot_regions=plot_regions)
    plt.show()

    return


def plot_read_current(
    ax: Axes,
    data_dict: dict,
) -> Axes:
    color_map = plt.get_cmap("RdBu")

    channel_critical_currents_mesh: np.ndarray = data_dict.get(
        "channel_critical_currents_mesh"
    )
    zero_state_currents: np.ndarray = data_dict.get("zero_state_currents")
    one_state_currents: np.ndarray = data_dict.get("one_state_currents")
    zero_state_currents_inv: np.ndarray = data_dict.get("zero_state_currents_inv")
    one_state_currents_inv: np.ndarray = data_dict.get("one_state_currents_inv")
    read_currents_mesh: np.ndarray = data_dict.get("read_currents_mesh")

    inv_region = np.where(
        (zero_state_currents_inv < read_currents_mesh)
        * (read_currents_mesh < one_state_currents_inv),
        read_currents_mesh,
        np.nan,
    )
    nominal_region = np.where(
        (read_currents_mesh > one_state_currents)
        * (read_currents_mesh < zero_state_currents),
        read_currents_mesh,
        np.nan,
    )

    c1 = ax.pcolormesh(
        channel_critical_currents_mesh,
        read_currents_mesh,
        nominal_region,
        cmap=color_map,
        vmin=-1000,
        vmax=1000,
    )
    c2 = ax.pcolormesh(
        channel_critical_currents_mesh,
        read_currents_mesh,
        -1 * inv_region,
        cmap=color_map,
        vmin=-1000,
        vmax=1000,
    )
    ax.set_xlabel("Channel Critical Current ($I_{C, CH}(I_{RE})$)) [uA]")
    ax.set_ylabel("Set Read Current [uA]")
    # ax.legend(["Nominal Region", "Inverting Region"])
    cbar = plt.colorbar(c2)
    cbar.set_label("Signed Read Current [uA]")

    return ax


def plot_region_dict(ax: Axes, regions: dict, x: np.ndarray, y: np.ndarray) -> Axes:
    color_cycler = cycler(colors=["red", "orange", "magenta", "skyblue"])
    for (name, region), colors in zip(regions.items(), color_cycler):
        region_plot = plt.contour(
            x,
            y,
            region,
            levels=[0.5, 1.5],
            **colors,
        )

        ax.clabel(
            region_plot,
            inline=True,
            fontsize=14,
            fmt=name,
            inline_spacing=0,
            rightside_up=True,
        )
    return ax


def plot_edge_fits(ax: Axes, critical_currents: np.ndarray, lines: list) -> Axes:
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for line in lines:
        ax = plot_edge_fit(ax, critical_currents, **line)
        print(line)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return ax


def plot_edge_fit(ax: Axes, x: np.ndarray, p1: float, p2: float) -> Axes:
    y = p1 * x + p2
    ax.plot(x, y, color="red")
    return ax


def plot_read_margin(
    ax: Axes,
    data_dict: dict,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    read_margins = data_dict["read_margins"]
    read_currents = data_dict["read_currents"]
    set_read_current = data_dict["set_read_current"]
    width_ratio = data_dict["width_ratio"]

    ax.pcolor(
        left_critical_currents_mesh,
        write_currents_mesh,
        read_margins,
        linewidth=0.5,
        shading="auto",
    )

    ax.set_xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    ax.set_ylabel("Write Current [uA]")
    ax.set_title("Read Currents Margins")
    ax.invert_xaxis()
    cbar = plt.colorbar()

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ic*width_ratio:.0f}" for ic in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    inv_region = ax.contour(
        left_critical_currents_mesh,
        write_currents_mesh,
        data_dict["regions"]["inverting"],
        levels=[0],
        colors="white",
        linestyles="dashed",
    )
    ax.clabel(
        inv_region,
        inline=True,
        fontsize=10,
        fmt="Inverting",
        inline_spacing=0,
        rightside_up=True,
    )

    return ax
