import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
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
    plt.pcolormesh(xx, yy, ber)
    # plt.gca().invert_xaxis()
    plt.xlabel("Enable Current [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("BER vs Write Current and Critical Current")
    cbar = plt.colorbar()
    plt.clim(0, 1)
    cbar.set_ticks([0, 0.5, 1])
    return ax


def plot_htron_sweep_scaled(
    ax: plt.Axes,
    left_critical_currents: np.ndarray,
    write_currents: np.ndarray,
    ber: np.ndarray,
):
    plt.sca(ax)
    xx, yy = np.meshgrid(left_critical_currents, write_currents)
    plt.pcolor(xx, yy, ber, shading="auto")
    plt.xlabel("Left Critical Current [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("BER vs Write Current and Critical Current")
    cbar = plt.colorbar()
    plt.clim(0, 1)
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
    ax: plt.Axes,
    data_dict: dict,
    plot_regions=False,
    data_point: tuple = None,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    width_ratio = data_dict["width_ratio"]

    total_persistent_current, regions = calculate_persistent_current(data_dict)
    regions = {"voltage": regions["voltage"]}
    c = plt.pcolormesh(
        left_critical_currents_mesh,
        write_currents_mesh,
        total_persistent_current,
        edgecolors="none",
        linewidth=0.5,
    )

    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    # plt.title("Maximum Persistent Current")
    plt.gca().invert_xaxis()
    cbar = plt.colorbar()
    cbar.set_label("Maximum Persistent Current [uA]")
    if plot_regions:
        color_cycler = cycler(colors=["magenta", "skyblue"])
        for (name, region), colors in zip(regions.items(), color_cycler):
            region_plot = plt.contour(
                left_critical_currents_mesh,
                write_currents_mesh,
                region,
                levels=[0.5, 1.5],
                **colors,
            )

            plt.clabel(
                region_plot,
                inline=True,
                fontsize=14,
                fmt=name,
                inline_spacing=0,
                rightside_up=True,
            )

    if data_point is not None:
        ax = plot_point(ax, *data_point, marker="*", color="red", markersize=15)

    # ax.set_xlim(right=0)

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ichl*width_ratio:.0f}" for ichl in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    data_dict["persistent_currents"] = total_persistent_current
    data_dict["regions"] = regions

    return ax, data_dict


def plot_zero_state_currents(
    ax: plt.Axes,
    data_dict: dict,
    data_point: tuple = None,
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
    )

    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("Zero State Current")
    plt.gca().invert_xaxis()
    cbar = plt.colorbar()
    cbar.set_ticks([cbar.vmin, cbar.vmax])

    if data_point is not None:
        ax = plot_point(ax, *data_point, marker="*", color="red", markersize=15)

    # ax.set_xlim(right=0)

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
    data_point: tuple = None,
    plot_regions=False,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    width_ratio = data_dict["width_ratio"]

    one_state_currents = data_dict["one_state_currents"]

    c = plt.pcolormesh(
        left_critical_currents_mesh,
        write_currents_mesh,
        one_state_currents,
        edgecolors="none",
        linewidth=0.5,
    )

    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("One State Current")
    plt.gca().invert_xaxis()
    cbar = plt.colorbar()
    cbar.set_ticks([cbar.vmin, cbar.vmax])

    if data_point is not None:
        ax = plot_point(ax, *data_point, marker="*", color="red", markersize=15)

    # ax.set_xlim(right=0)

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


def plot_state_currents(data_dict: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.sca(axes[0])
    axes[0] = plot_zero_state_currents(axes[0], data_dict, plot_regions=True)
    plt.sca(axes[1])
    axes[1] = plot_one_state_currents(axes[1], data_dict, plot_regions=True)
    plt.show()


def plot_read_current(
    ax: plt.Axes,
    data_dict: dict,
    data_point: tuple = None,
    contour: bool = False,
    plot_regions: bool = False,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    persistent_currents = data_dict["persistent_currents"]
    width_ratio = data_dict["width_ratio"]
    set_read_current = data_dict["set_read_current"]

    read_current_dict = calculate_read_currents(data_dict)
    read_currents = read_current_dict["read_currents"]
    read_margins = read_current_dict["read_margins"]
    read_margins = np.where(read_currents < write_currents_mesh, 0, read_margins)
    read_margins = np.where(persistent_currents == 0, 0, read_margins)
    read_margins = np.where(
        read_currents < read_current_dict["zero_state_currents"], 0, read_margins
    )
    plt.pcolormesh(
        left_critical_currents_mesh,
        write_currents_mesh,
        read_margins,
        linewidth=0.5,
    )
    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Set Read Current [uA]")
    plt.title("Ideal Read Current Margin")
    plt.gca().invert_xaxis()
    cbar = plt.colorbar()
    cbar.set_label("Read Current Margin [uA]")
    # ax.set_xlim(right=0)

    if contour:
        # Add contour at set_read_current
        operating_regions = np.where(
            (set_read_current > read_current_dict["zero_state_currents"])
            * (set_read_current < read_current_dict["one_state_currents"]),
            1,
            0,
        )
        operating_regions = np.where(read_margins == 0, 0, operating_regions)

        inverting_region = np.where(
            (set_read_current < read_current_dict["zero_state_currents"])
            * (set_read_current > read_current_dict["one_state_currents"]),
            1,
            0,
        )
        operating_margin = plt.contourf(
            left_critical_currents_mesh,
            write_currents_mesh,
            operating_regions,
            levels=[0.5, 1.5],
            colors="white",
            alpha=0.5,
        )
        plt.contour(
            left_critical_currents_mesh,
            write_currents_mesh,
            operating_regions,
            levels=[0.5, 1.5],
            colors="white",
            linestyles="solid",
        )

        plt.contour(
            left_critical_currents_mesh,
            write_currents_mesh,
            read_current_dict["read_margins"],
            levels=[set_read_current - 10, set_read_current, set_read_current + 10],
            colors="white",
            linestyles="dashed",
        )

        inv_region = plt.contour(
            left_critical_currents_mesh,
            write_currents_mesh,
            inverting_region,
            levels=[0.5, 1.5],
            colors="black",
            linestyles="dashed",
        )
        plt.clabel(
            inv_region,
            inline=True,
            fontsize=10,
            fmt="Inverting",
            inline_spacing=0,
            rightside_up=True,
        )

    if data_point is not None:
        ax = plot_point(ax, *data_point, marker="*", color="red", markersize=15)

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ichl*width_ratio:.0f}" for ichl in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    ax = plt.gca()
    if plot_regions:
        ax = plot_region_dict(
            ax, data_dict["regions"], left_critical_currents_mesh, write_currents_mesh
        )

    data_dict["read_currents"] = read_currents
    data_dict["read_margins"] = read_current_dict["read_margins"]
    return ax


def plot_region_dict(ax, regions: dict, x, y):
    plt.sca(ax)
    color_cycler = cycler(colors=["red", "orange", "magenta", "skyblue"])
    for (name, region), colors in zip(regions.items(), color_cycler):
        region_plot = plt.contour(
            x,
            y,
            region,
            levels=[0.5, 1.5],
            **colors,
        )

        plt.clabel(
            region_plot,
            inline=True,
            fontsize=14,
            fmt=name,
            inline_spacing=0,
            rightside_up=True,
        )
    ax = plt.gca()
    return ax


def plot_edge_fits(ax, lines, critical_currents):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for line in lines:
        ax = plot_edge_fit(ax, critical_currents, **line)
        print(line)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return ax


def plot_edge_fit(ax, x, p1, p2):
    y = p1 * x + p2
    ax.plot(x, y, color="red")
    return ax


def plot_read_margin(
    ax: plt.Axes,
    data_dict: dict,
    data_point: tuple = None,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    read_margins = data_dict["read_margins"]
    read_currents = data_dict["read_currents"]
    set_read_current = data_dict["set_read_current"]
    width_ratio = data_dict["width_ratio"]

    plt.pcolor(
        left_critical_currents_mesh,
        write_currents_mesh,
        read_margins,
        linewidth=0.5,
        shading="auto",
    )

    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("Read Currents Margins")
    plt.gca().invert_xaxis()
    cbar = plt.colorbar()

    if data_point is not None:
        ax = plot_point(ax, *data_point, marker="*", color="red", markersize=15)

    # ax.set_xlim(right=0)

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ic*width_ratio:.0f}" for ic in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    inv_region = plt.contour(
        left_critical_currents_mesh,
        write_currents_mesh,
        data_dict["regions"]["inverting"],
        levels=[0],
        colors="white",
        linestyles="dashed",
    )
    plt.clabel(
        inv_region,
        inline=True,
        fontsize=10,
        fmt="Inverting",
        inline_spacing=0,
        rightside_up=True,
    )
