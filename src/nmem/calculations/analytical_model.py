import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

from nmem.calculations.calculations import (
    calculate_ideal_read_current,
    calculate_ideal_read_margin,
    calculate_left_branch_current,
    calculate_left_lower_bound,
    calculate_left_upper_bound,
    calculate_one_state_current,
    calculate_right_branch_current,
    calculate_right_lower_bound,
    calculate_right_upper_bound,
    calculate_zero_state_current,
    htron_critical_current,
)
from nmem.calculations.plotting import (
    plot_htron_sweep,
    plot_persistent_current,
    plot_point,
    plot_read_current,
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
    max_critical_current = data_dict["max_critical_current"]
    # Initial Write
    left_branch_write_current = calculate_left_branch_current(alpha, write_current, 0)
    right_branch_write_current = calculate_right_branch_current(alpha, write_current, 0)

    # Set Read
    zero_state_current = calculate_zero_state_current(
        left_critical_current,
        right_critical_current,
        persistent_current,
        alpha,
        iretrap,
        max_critical_current,
    )
    one_state_current = calculate_one_state_current(
        left_critical_current,
        right_critical_current,
        persistent_current,
        alpha,
        iretrap,
        max_critical_current,
    )

    ideal_read_current = calculate_ideal_read_current(
        zero_state_current, one_state_current
    )
    ideal_read_margin = calculate_ideal_read_margin(
        zero_state_current, one_state_current
    )

    noninverting_operation = zero_state_current < one_state_current

    min_ichl = calculate_left_lower_bound(persistent_current, set_read_current, alpha)
    max_ichl = calculate_left_upper_bound(persistent_current, set_read_current, alpha)
    min_ichr = calculate_right_lower_bound(persistent_current, set_read_current, alpha)
    max_ichr = calculate_right_upper_bound(persistent_current, set_read_current, alpha)

    ichl_within_range = min_ichl < left_critical_current < max_ichl
    ichr_within_range = min_ichr < right_critical_current < max_ichr

    set_param_dict = {
        "Total Critical Current (enable off) [uA]": f"{HTRON_INTERCEPT:.2f}",
        "Left Critical Current (enable off) [uA]": f"{max_left_critical_current:.2f}",
        "Right Critical Current (enable off) [uA]": f"{max_right_critical_current:.2f}",
        "Width Ratio": f"{width_ratio:.2f}",
        "Inductive Ratio (alpha)": f"{alpha:.2f}",
        "Retrap Ratio": f"{iretrap:.2f}",
    }
    write_param_dict = {
        "Write Current [uA]": f"{write_current:.2f}",
        "Enable Write Current [uA]": f"{enable_write_current:.2f}",
        "Left Branch Write Current [uA]": f"{left_branch_write_current:.2f}",
        "Right Branch Write Current [uA]": f"{right_branch_write_current:.2f}",
        "Left Side Critical Current (enable on) [uA]": f"{left_critical_current:.2f}",
        "Right Side Critical Current (enable on) [uA]": f"{right_critical_current:.2f}",
        "Left Side Retrapping Current (enable on) [uA]": f"{left_critical_current*iretrap:.2f}",
        "Right Side Retrapping Current (enable on) [uA]": f"{right_critical_current*iretrap:.2f}",
        "Maximum Persistent Current": f"{persistent_current:.2f}",
    }
    read_param_dict = {
        "Left Side Critical Current (enable on) [uA]": f"{left_critical_current:.2f}",
        "Right Side Critical Current (enable on) [uA]": f"{right_critical_current:.2f}",
        "Maximum Persistent Current": f"{persistent_current:.2f}",
        "Zero State Current [uA]": f"{zero_state_current:.2f}",
        "One State Current [uA]": f"{one_state_current:.2f}",
        "Set Read Current [uA]": f"{set_read_current:.2f}",
        "Ideal Read Current [uA]": f"{ideal_read_current:.2f}",
        "Ideal Read Margin [uA]": f"{ideal_read_margin:.2f}",
        "Non-inverting Operation": f"{noninverting_operation}",
        "ICHL within range": f"{ichl_within_range}",
        "ICHR within range": f"{ichr_within_range}",
        "minimum ICHL [uA] (set read)": f"{min_ichl:.2f}",
        "maximum ICHL [uA] (set read)": f"{max_ichl:.2f}",
        "minimum ICHR [uA] (set read)": f"{min_ichr:.2f}",
        "maximum ICHR [uA] (set read)": f"{max_ichr:.2f}",
    }
    param_dict = {
        # **set_param_dict,
        # **write_param_dict,
        **read_param_dict,
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
    read_currents_new = np.where(
        (read_currents < (set_read_current + read_margins))
        & (read_currents > (set_read_current - read_margins)),
        read_currents,
        0,
    )

    plt.pcolor(
        left_critical_currents_mesh,
        write_currents_mesh,
        read_currents_new,
        linewidth=0.5,
        shading="auto",
    )

    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("Read Margin")
    plt.gca().invert_xaxis()
    cbar = plt.colorbar()
    # cbar.set_ticks(np.linspace(np.min(read_margins), np.max(read_margins), 5))

    # ax.set_xlim(right=0)

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ic*width_ratio:.0f}" for ic in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    return


if __name__ == "__main__":
    # HTRON_SLOPE = -2.69  # uA / uA
    # HTRON_INTERCEPT = 1257  # uA
    HTRON_SLOPE = -4.371  # uA / uA
    HTRON_INTERCEPT = 1726.201  # uA
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.3
    ALPHA = 1 - 0.3

    MAX_CRITICAL_CURRENT = 830
    IRETRAP = 0.2
    IREAD = 156
    IDXX = 30
    IDXY = 35
    # IDXX = 20
    # IDXY = 20
    N = 50
    FILE_PATH = "/home/omedeiro/"
    FILE_NAME = "SPG806_20240804_nMem_ICE_ber_D6_A4_2024-08-04 16-38-50.mat"
    EDGE_FITS = [
        {"p1": 2.818, "p2": -226.1},
        {"p1": 2.818, "p2": -165.1},
        {"p1": 6.272, "p2": -433.9},
        {"p1": 6.272, "p2": -353.8},
    ]

    width_ratio = WIDTH_RIGHT / WIDTH_LEFT
    max_left_critical_current = HTRON_INTERCEPT / width_ratio
    max_right_critical_current = HTRON_INTERCEPT

    matlab_data_dict = import_matlab_data(FILE_PATH + "/" + FILE_NAME)

    # Extract write enable sweep data.
    enable_write_currents_measured = np.transpose(matlab_data_dict["x"][:, :, 1]) * 1e6
    write_currents_measured = np.transpose(matlab_data_dict["y"][:, :, 1]) * 1e6
    ber = matlab_data_dict["ber"]
    ber_2D = np.reshape(
        ber,
        (len(write_currents_measured), len(enable_write_currents_measured)),
        order="F",
    )

    # Define the write and enable write currents
    enable_write_currents = np.linspace(300, 400, N)
    write_currents = np.linspace(
        write_currents_measured[0], write_currents_measured[-1], N
    )

    # Calculate the channel critical current
    channel_current_enabled = htron_critical_current(
        enable_write_currents,
        HTRON_SLOPE,
        HTRON_INTERCEPT,
    )
    channel_current_enabled_measured = htron_critical_current(
        enable_write_currents_measured,
        HTRON_SLOPE,
        HTRON_INTERCEPT,
    )

    # Define the critical currents for the left and right branches
    left_critical_currents_measured = channel_current_enabled_measured / width_ratio
    right_critical_currents_measured = channel_current_enabled_measured

    left_critical_currents = channel_current_enabled / width_ratio
    right_critical_currents = channel_current_enabled

    # Create the meshgrid for the critical currents
    [left_critical_currents_mesh, write_currents_mesh] = np.meshgrid(
        left_critical_currents, write_currents
    )
    right_critical_currents_mesh = left_critical_currents_mesh * width_ratio
    # Create the data dictionary
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
        "max_critical_current": MAX_CRITICAL_CURRENT,
        "max_left_critical_current": max_left_critical_current,
        "max_right_critical_current": max_right_critical_current,
    }

    # Plot experimental data
    fig, ax = plt.subplots()
    ax = plot_htron_sweep(
        ax, write_currents_measured, enable_write_currents_measured, ber_2D
    )
    ax = plot_point(
        ax,
        enable_write_currents_measured[
            int(IDXX / N * len(enable_write_currents_measured))
        ],
        write_currents_measured[int(IDXY / N * len(write_currents_measured))],
        marker="*",
        color="red",
        markersize=15,
    )
    # ax.invert_xaxis()

    # Plot the persistent current
    fig, ax = plt.subplots()
    ax, max_persistent_currents, regions = plot_persistent_current(
        ax,
        data_dict,
        plot_regions=True,
    )
    data_dict["persistent_currents"] = max_persistent_currents
    data_dict["inverting_region"] = regions["both_switch"]
    data_dict["noninverting_region"] = regions["left_switch"]

    ax = plot_point(
        ax,
        left_critical_currents[IDXX],
        write_currents[IDXY],
        marker="*",
        color="red",
        markersize=15,
    )

    # Plot the read current
    fig, ax = plt.subplots()
    ax, read_currents, read_margins = plot_read_current(
        ax,
        data_dict,
    )
    data_dict["read_currents"] = read_currents
    data_dict["read_margins"] = read_margins

    zero_state_currents = calculate_zero_state_current(
        left_critical_currents_mesh,
        right_critical_currents_mesh,
        max_persistent_currents,
        ALPHA,
        IRETRAP,
        MAX_CRITICAL_CURRENT,
    )
    one_state_currents = calculate_one_state_current(
        left_critical_currents_mesh,
        right_critical_currents_mesh,
        max_persistent_currents,
        ALPHA,
        IRETRAP,
        MAX_CRITICAL_CURRENT,
    )
    data_dict["zero_state_currents"] = zero_state_currents
    data_dict["one_state_currents"] = one_state_currents
    ax = plot_point(
        ax,
        left_critical_currents[IDXX],
        write_currents[IDXY],
        marker="*",
        color="red",
        markersize=15,
    )

    data_dict["ideal_read_margins"] = calculate_ideal_read_margin(
        zero_state_currents, one_state_currents
    )

    # Plot the read margin
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

    plt.show()

    # Get the point parameters
    param_df = get_point_parameters(IDXX, IDXY, data_dict)
    print(param_df)
