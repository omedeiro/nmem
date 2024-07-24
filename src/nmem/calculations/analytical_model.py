import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from nmem.calculations.calculations import (
    calculate_0_state_currents,
    calculate_1_state_currents,
    calculate_left_branch_current,
    calculate_right_branch_current,
    htron_critical_current,
)
from nmem.calculations.plotting import (
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
    persistent_current = float(data_dict["persistent_currents"][i, j])
    max_left_critical_current = data_dict["max_left_critical_current"]
    max_right_critical_current = data_dict["max_right_critical_current"]

    alpha = data_dict["alpha"]
    ic_ratio = data_dict["ic_ratio"]
    width_ratio = data_dict["width_ratio"]
    
    # Initial Write
    left_branch_write_current = calculate_left_branch_current(alpha, write_current, 0)
    right_branch_write_current = calculate_right_branch_current(alpha, write_current, 0)

    # ideal_read_current = np.mean([zero_current_left, zero_current_left])
    # ideal_read_margin = (zero_current_left - zero_current_left) / 2

    param_dict = {
        "Total Critical Current (enable off) [uA]": f"{IC0:.2f}",
        "Left Critical Current (enable off) [uA]": f"{max_left_critical_current:.2f}",
        "Right Critical Current (enable off) [uA]": f"{max_right_critical_current:.2f}",
        "Width Ratio": f"{width_ratio:.2f}",
        "Inductive Ratio": f"{ALPHA:.2f}",
        "Switching Current Ratio": f"{ic_ratio:.2f}",
        "Write Current [uA]": f"{write_current:.2f}",
        "Enable Write Current [uA]": f"{enable_write_current:.2f}",
        "Left Branch Write Current [uA]": f"{left_branch_write_current:.2f}",
        "Right Branch Write Current [uA]": f"{right_branch_write_current:.2f}",
        "Left Side Critical Current (enable on) [uA]": f"{left_critical_current:.2f}",
        "Right Side Critical Current (enable on) [uA]": f"{right_critical_current:.2f}",
        "Persistent Current": f"{persistent_current:.2f}",
        # "Zero State Left Branch Current [uA]": f"{zero_current_left:.2f}",
        # "Zero State Right Branch Current [uA]": f"{zero_current_right:.2f}",
        # "One State Left Branch Current [uA]": f"{one_current_left:.2f}",
        # "One State Right Branch Current [uA]": f"{one_current_right:.2f}",
        "Set Read Current [uA]": f"{IREAD:.2f}",
        # "Ideal Read Current [uA]": f"{ideal_read_current:.2f}",
        # "Ideal Read Margin [uA]": f"{ideal_read_margin:.2f}",
    }
    param_df = pd.DataFrame(param_dict.values(), index=param_dict.keys())
    param_df.columns = ["Value"]

    return param_df


def plot_read_margin(
    ax: plt.Axes,
    left_switching_currents: np.ndarray,
    write_currents: np.ndarray,
    read_currents: np.ndarray,
    persistent_currents: np.ndarray,
    alpha: float,
    iretrap: float,
    set_read_current: float,
):
    [xx, yy] = np.meshgrid(left_switching_currents, write_currents)
    right_critical_currents = left_switching_currents / ic_ratio

    zero_current_left, zero_current_right = calculate_0_state_currents(
        left_critical_currents,
        right_critical_currents,
        persistent_currents,
        alpha,
    )
    one_current_left, one_current_right = calculate_1_state_currents(
        left_critical_currents,
        right_critical_currents,
        persistent_currents,
        alpha,
    )
    read_currents = (zero_current_right + one_current_right) / 2
    # inverting_operation = zero_current_right < one_current_right
    plt.pcolormesh(xx, yy, read_currents, linewidth=0.5)
    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    # plt.title("Read Margin")
    plt.gca().invert_xaxis()
    plt.colorbar()
    ax.set_xlim(right=0)

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ic/ic_ratio:.0f}" for ic in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    return


if __name__ == "__main__":
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.4
    IC0 = 600
    HTRON_SLOPE = -2.69
    HTRON_INTERCEPT = 1000
    ALPHA = 1 - (1 / 2.818)
    BETA = 0.159
    IRETRAP = 1
    IREAD = 280
    IDXX = 25
    IDXY = 5
    N = 50
    FILE_PATH = "/home/omedeiro/"
    FILE_NAME = (
        "SPG806_nMem_ICE_writeEnable_sweep_square11_D6_D1_2023-12-13 02-06-48.mat"
    )
    width_ratio = WIDTH_LEFT / (WIDTH_LEFT + WIDTH_RIGHT)
    max_left_critical_current = IC0 * width_ratio
    max_right_critical_current = IC0 * (1 - width_ratio)

    ic_ratio = max_left_critical_current / max_right_critical_current

    matlab_data_dict = import_matlab_data(FILE_PATH + "/" + FILE_NAME)

    # Extract write enable sweep data.
    enable_write_currents = np.transpose(matlab_data_dict["x"][:, :, 1]) * 1e6
    write_currents = np.transpose(matlab_data_dict["y"][:, :, 1]) * 1e6
    ber = matlab_data_dict["ber"]
    ber_2D = np.reshape(ber, (len(write_currents), len(enable_write_currents)))

    # Plot experimental data
    # fig, ax = plt.subplots()
    # ax = plot_htron_sweep(ax, write_currents, enable_write_currents, ber_2D)

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

    left_critical_currents = channel_critical_current_enabled * ic_ratio
    right_critical_currents = channel_critical_current_enabled * (1 - ic_ratio)

    [left_critical_currents_mesh, write_critical_currents_mesh] = np.meshgrid(
        left_critical_currents, write_currents
    )
    data_dict = {
        "left_critical_currents": left_critical_currents,
        "right_critical_currents": right_critical_currents,
        "left_critical_currents_mesh": left_critical_currents_mesh,
        "write_currents": write_currents,
        "write_currents_mesh": write_critical_currents_mesh,
        "enable_write_currents": enable_write_currents,
        "ber": ber_2D,
        "alpha": ALPHA,
        "iretrap": IRETRAP,
        "iread": IREAD,
        "width_left": WIDTH_LEFT,
        "width_right": WIDTH_RIGHT,
        "width_ratio": width_ratio,
        "max_channel_critical_current": IC0,
        "max_left_critical_current": max_left_critical_current,
        "max_right_critical_current": max_right_critical_current,
        "ic_ratio": ic_ratio,
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

    # fig, ax = plt.subplots()
    # ax, read_currents = plot_read_current(
    #     ax,
    #     left_critical_currents,
    #     write_currents,
    #     total_persistent_current,
    #     ALPHA,
    #     ICHL,
    #     ICHR,
    #     IRETRAP,
    #     plot_region=False,
    # )

    #     ax = plot_point(
    #         ax,
    #         left_critical_currents[IDXX],
    #         write_currents[IDXY],
    #         marker="*",
    #         color="red",
    #         markersize=15,
    #     )

    #     plt.show()

    param_df = get_point_parameters(IDXX, IDXY, data_dict)
    print(param_df)

    #     # %%
    fig, ax = plt.subplots()
    read_margin = plot_read_margin(
        ax,
        left_critical_currents,
        write_currents,
        np.zeros_like(write_currents),
        total_persistent_current,
        ALPHA,
        IRETRAP,
        IREAD,
    )
    ax = plot_point(
        ax,
        left_critical_currents[IDXX],
        write_currents[IDXY],
        marker="*",
        color="red",
        markersize=15,
    )
