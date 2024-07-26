import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from nmem.calculations.calculations import (
    calculate_channel_current_one,
    calculate_channel_current_zero,
    calculate_left_branch_current,
    calculate_one_state_current,
    calculate_right_branch_current,
    calculate_zero_state_current,
    htron_critical_current,
)
from nmem.calculations.plotting import (
    plot_htron_sweep,
    plot_persistent_current,
    plot_point,
    plot_read_current,
    plot_htron_sweep_scaled,
)


def import_matlab_data(file_path):
    data = sio.loadmat(file_path)
    return data


def get_point_parameters(i: int, j: int, data_dict: dict):
    left_critical_current = float(data_dict["left_critical_currents"][i])
    right_critical_current = float(data_dict["right_critical_currents"][i])
    write_current = float(data_dict["write_currents"][j])
    enable_write_current = float(data_dict["enable_write_currents"][i])
    persistent_currents = data_dict["persistent_currents"]
    persistent_current = float(data_dict["persistent_currents"][j, i])
    max_left_critical_current = data_dict["max_left_critical_current"]
    max_right_critical_current = data_dict["max_right_critical_current"]
    iretrap = data_dict["iretrap"]
    alpha = data_dict["alpha"]
    width_ratio = data_dict["width_ratio"]
    set_read_current = data_dict["set_read_current"]
    # Initial Write
    left_branch_write_current = calculate_left_branch_current(alpha, write_current, 0)
    right_branch_write_current = calculate_right_branch_current(alpha, write_current, 0)

    zero_currents_left, zero_currents_right = calculate_channel_current_zero(
        left_critical_current, right_critical_current, persistent_current, alpha
    )

    one_currents_left, one_currents_right = calculate_channel_current_one(
        left_critical_current, right_critical_current, persistent_current, alpha
    )
    # Zero State
    zero_state_current = calculate_zero_state_current(
        left_critical_currents_mesh[j, i],
        right_critical_currents_mesh[j, i],
        persistent_currents[j, i],
        alpha,
        iretrap,
    )
    one_state_current = calculate_one_state_current(
        left_critical_currents_mesh[j, i],
        right_critical_currents_mesh[j, i],
        persistent_currents[j, i],
        alpha,
        iretrap,
    )

    left_retrap_current = left_critical_current + right_critical_current * iretrap
    right_retrap_current = right_critical_current + left_critical_current * iretrap

    ideal_read_current = (zero_state_current + one_state_current) / 2
    ideal_read_margin = np.abs(zero_state_current - one_state_current) / 2
    inverting_operation = zero_state_current < one_state_current

    param_dict = {
        "Total Critical Current (enable off) [uA]": f"{HTRON_INTERCEPT:.2f}",
        "Left Critical Current (enable off) [uA]": f"{max_left_critical_current:.2f}",
        "Right Critical Current (enable off) [uA]": f"{max_right_critical_current:.2f}",
        "Width Ratio": f"{width_ratio:.2f}",
        "Inductive Ratio": f"{alpha:.2f}",
        "Retrap Ratio": f"{iretrap:.2f}",
        "Write Current [uA]": f"{write_current:.2f}",
        "Enable Write Current [uA]": f"{enable_write_current:.2f}",
        "Left Branch Write Current [uA]": f"{left_branch_write_current:.2f}",
        "Right Branch Write Current [uA]": f"{right_branch_write_current:.2f}",
        "Left Side Critical Current (enable on) [uA]": f"{left_critical_currents_mesh[j, i]:.2f}",
        "Right Side Critical Current (enable on) [uA]": f"{right_critical_currents_mesh[j, i]:.2f}",
        "Persistent Current": f"{ persistent_currents[j, i]:.2f}",
        "Zero Current Left [uA]": f"{zero_currents_left:.2f}",
        "Zero Current Right [uA]": f"{zero_currents_right:.2f}",
        "One Current Left [uA]": f"{one_currents_left:.2f}",
        "One Current Right [uA]": f"{one_currents_right:.2f}",
        "Left Retrap Current [uA]": f"{left_retrap_current:.2f}",
        "Right Retrap Current [uA]": f"{right_retrap_current:.2f}",
        "State 0 Current [uA]": f"{zero_state_current:.2f}",
        "State 1 Current [uA]": f"{one_state_current:.2f}",
        "Set Read Current [uA]": f"{set_read_current:.2f}",
        "Ideal Read Current [uA]": f"{ideal_read_current:.2f}",
        "Ideal Read Margin [uA]": f"{ideal_read_margin:.2f}",
        "Inverting Operation": f"{inverting_operation}",
    }
    param_df = pd.DataFrame(param_dict.values(), index=param_dict.keys())
    param_df.columns = ["Value"]

    return param_df


def plot_read_margin(
    ax: plt.Axes,
    data_dict: dict,
):
    left_critical_currents_mesh = data_dict["left_critical_currents_mesh"]
    write_currents_mesh = data_dict["write_currents_mesh"]
    read_margins = data_dict["read_margins"]
    read_currents = data_dict["read_currents"]
    set_read_current = data_dict["set_read_current"]
    
    
    set_read_current_array = np.ones_like(read_margins) * set_read_current
    # read_currents_new = np.where(read_currents < set_read_current_array, 0, read_currents)

    plt.pcolormesh(
        left_critical_currents_mesh,
        write_currents_mesh,
        read_currents,
        linewidth=0.5,
        vmin=50,
        vmax=200,
    )
    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    # plt.title("Read Margin")
    plt.gca().invert_xaxis()
    plt.colorbar()
    # ax.set_xlim(right=0)

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels(
        [f"{ic*(1-width_ratio)/width_ratio:.0f}" for ic in ax.get_xticks()]
    )
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    return


if __name__ == "__main__":

    HTRON_SLOPE = -2.5  # uA / uA
    HTRON_INTERCEPT = 950  # uA
    WIDTH_LEFT = 0.10
    WIDTH_RIGHT = 0.30
    ALPHA = 1 - .33

    IRETRAP = 0.5
    IREAD = 200
    IDXX = 13
    IDXY = 20
    N = 50
    FILE_PATH = "/home/omedeiro/"
    FILE_NAME = (
        "SPG806_nMem_ICE_writeEnable_sweep_square11_D6_D1_2023-12-13 02-06-48.mat"
    )
    width_ratio = WIDTH_LEFT / (WIDTH_LEFT + WIDTH_RIGHT)
    max_left_critical_current = HTRON_INTERCEPT * width_ratio
    max_right_critical_current = HTRON_INTERCEPT * (1 - width_ratio)

    matlab_data_dict = import_matlab_data(FILE_PATH + "/" + FILE_NAME)

    # Extract write enable sweep data.
    enable_write_currents = np.transpose(matlab_data_dict["x"][:, :, 1]) * 1e6
    write_currents = np.transpose(matlab_data_dict["y"][:, :, 1]) * 1e6
    ber = matlab_data_dict["ber"]
    ber_2D = np.reshape(ber, (len(write_currents), len(enable_write_currents)))

    # Plot experimental data
    fig, ax = plt.subplots()
    ax_htron = plot_htron_sweep(ax, write_currents, enable_write_currents, ber_2D)

    EDGE_FITS = [
        {"p1": 2.818, "p2": -226.1},
        {"p1": 2.818, "p2": -165.1},
        {"p1": 6.272, "p2": -433.9},
        {"p1": 6.272, "p2": -353.8},
    ]

    enable_write_currents = np.linspace(
        enable_write_currents[0], enable_write_currents[-1], N
    )
    write_currents = np.linspace(write_currents[0], write_currents[-1], N)

    channel_critical_current_enabled = htron_critical_current(
        HTRON_SLOPE, HTRON_INTERCEPT, enable_write_currents
    )

    enable_write_currents_measured = np.linspace(
        enable_write_currents[0], enable_write_currents[-1], 23
    )
    write_currents_measured = np.linspace(write_currents[0], write_currents[-1], 21)

    channel_critical_current_enabled_measured = htron_critical_current(
        HTRON_SLOPE, HTRON_INTERCEPT, enable_write_currents_measured
    )
    left_critical_currents_measured = (
        channel_critical_current_enabled_measured * width_ratio
    )
    right_critical_currents_measured = channel_critical_current_enabled_measured * (
        1 - width_ratio
    )

    left_critical_currents = channel_critical_current_enabled * width_ratio
    right_critical_currents = channel_critical_current_enabled * (1 - width_ratio)

    [left_critical_currents_mesh, write_currents_mesh] = np.meshgrid(
        left_critical_currents, write_currents
    )
    right_critical_currents_mesh = (
        left_critical_currents_mesh * (1 - width_ratio) / width_ratio
    )

    data_dict = {
        "left_critical_currents": left_critical_currents,
        "right_critical_currents": right_critical_currents,
        "left_critical_currents_mesh": left_critical_currents_mesh,
        "right_critical_currents_mesh": right_critical_currents_mesh,
        "write_currents": write_currents,
        "write_currents_mesh": write_currents_mesh,
        "enable_write_currents": enable_write_currents,
        "ber": ber_2D,
        "alpha": ALPHA,
        "iretrap": IRETRAP,
        "set_read_current": IREAD,
        "width_left": WIDTH_LEFT,
        "width_right": WIDTH_RIGHT,
        "width_ratio": width_ratio,
        "max_channel_critical_current": HTRON_INTERCEPT,
        "max_left_critical_current": max_left_critical_current,
        "max_right_critical_current": max_right_critical_current,
    }

    fig, ax = plt.subplots()
    ax, total_persistent_current, regions = plot_persistent_current(
        ax,
        data_dict,
        plot_regions=False,
    )
    data_dict["persistent_currents"] = total_persistent_current

    ax = plot_point(
        ax,
        left_critical_currents[IDXX],
        write_currents[IDXY],
        marker="*",
        color="red",
        markersize=15,
    )
    # plot_edge_fits(ax, EDGE_FITS, left_critical_currents)

    fig, ax = plt.subplots()
    ax, read_currents, read_margins = plot_read_current(
        ax,
        data_dict,
    )
    data_dict["read_currents"] = read_currents
    data_dict["read_margins"] = read_margins

    ax = plot_point(
        ax,
        left_critical_currents[IDXX],
        write_currents[IDXY],
        marker="*",
        color="red",
        markersize=15,
    )

    # plt.show()

    param_df = get_point_parameters(IDXX, IDXY, data_dict)
    print(param_df)

    #     # %%
    fig, ax = plt.subplots()
    read_margin = plot_read_margin(
        ax,
        data_dict,
    )
    ax = plot_point(
        ax,
        left_critical_currents[IDXX],
        write_currents[IDXY],
        marker="*",
        color="red",
        markersize=15,
    )
    ax = plot_htron_sweep_scaled(
        ax, write_currents_measured, left_critical_currents_measured, ber_2D
    )
