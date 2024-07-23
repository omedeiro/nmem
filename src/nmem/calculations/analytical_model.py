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
)


def import_matlab_data(file_path):
    data = sio.loadmat(file_path)
    return data


def get_point_parameters(
    i: int,
    j: int,
    critical_currents: np.ndarray,
    write_currents: np.ndarray,
    persistent_currents: np.ndarray,
    enable_currents: np.ndarray,
    alpha: float,
    iretrap: float,
):
    critical_current = float(critical_currents[i])
    write_current = float(write_currents[j])
    persistent_current = float(persistent_currents[j, i])
    enable_current = float(enable_currents[i])

    left_branch_write_current = calculate_left_branch_current(alpha, write_current, 0)
    right_branch_write_current = calculate_right_branch_current(alpha, write_current, 0)

    left_branch_current1 = calculate_left_branch_current(
        alpha, write_current, persistent_current
    )
    right_branch_current1 = calculate_right_branch_current(
        alpha, write_current, persistent_current
    )

    zero_current_left, zero_current_right = calculate_0_state_currents(
        critical_current,
        critical_current / IC_RATIO,
        persistent_current,
        alpha,
    )
    one_current_left, one_current_right = calculate_1_state_currents(
        critical_current,
        critical_current / IC_RATIO,
        persistent_current,
        alpha,
    )

    ideal_read_current = np.mean([zero_current_left, zero_current_left])
    ideal_read_margin = (zero_current_left - zero_current_left) / 2

    param_dict = {
        "Total Critical Current (enable off) [uA]": f"{IC0:.2f}",
        "Left Critical Current (enable off) [uA]": f"{ICHL:.2f}",
        "Right Critical Current (enable off) [uA]": f"{ICHR:.2f}",
        "Width Ratio": f"{WIDTH_RATIO:.2f}",
        "Inductive Ratio": f"{ALPHA:.2f}",
        "Switching Current Ratio": f"{IC_RATIO:.2f}",
        "Write Current [uA]": f"{write_current:.2f}",
        "Enable Write Current [uA]": f"{enable_current:.2f}",
        "Left Branch Write Current [uA]": f"{left_branch_write_current:.2f}",
        "Right Branch Write Current [uA]": f"{right_branch_write_current:.2f}",
        "Left Side Critical Current (enable on) [uA]": f"{critical_current:.2f}",
        "Right Side Critical Current (enable on) [uA]": f"{critical_current / IC_RATIO:.2f}",
        "Persistent Current": f"{persistent_current:.2f}",
        "Zero State Left Branch Current [uA]": f"{zero_current_left:.2f}",
        "Zero State Right Branch Current [uA]": f"{zero_current_right:.2f}",
        "One State Left Branch Current [uA]": f"{one_current_left:.2f}",
        "One State Right Branch Current [uA]": f"{one_current_right:.2f}",
        "Set Read Current [uA]": f"{IREAD:.2f}",
        "Ideal Read Current [uA]": f"{ideal_read_current:.2f}",
        "Ideal Read Margin [uA]": f"{ideal_read_margin:.2f}",
    }
    param_df = pd.DataFrame(param_dict.values(), index=param_dict.keys())
    param_df.columns = ["Value"]

    return param_df


# def persistent_bounds(
#         left_critical_currents: np.ndarray,
#         write_currents: np.ndarray,
#         alpha: float,
#         persistent_currents: np.ndarray,
# )


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
    right_critical_currents = left_switching_currents / IC_RATIO

    zero_current_left, zero_current_right = calculate_0_state_currents(
        left_critical_currents,
        right_critical_currents,
        persistent_currents,
        alpha,
        iretrap,
    )
    one_current_left, one_current_right = calculate_1_state_currents(
        left_critical_currents,
        right_critical_currents,
        persistent_currents,
        alpha,
        iretrap,
    )
    read_margin = zero_current_left

    read_margin = np.where(persistent_currents == 0, 0, read_margin)

    plt.pcolormesh(xx, yy, read_margin, linewidth=0.5)
    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("Read Margin")
    plt.gca().invert_xaxis()
    plt.colorbar()
    ax.set_xlim(right=0)

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticklabels([f"{ic/IC_RATIO:.0f}" for ic in ax.get_xticks()])
    ax2.set_xlabel("Right Branch Critical Current ($I_{C, H_R}(I_{RE})$) [uA]")

    return read_margin


if __name__ == "__main__":
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.4
    IC0 = 400
    HTRON_SLOPE = -2.69
    HTRON_INTERCEPT = 1000
    ALPHA = 1 - (-1 / HTRON_SLOPE)
    BETA = 0.159
    IRETRAP = 1
    IREAD = 280
    IDXX = 22
    IDXY = 5
    WIDTH_RATIO = WIDTH_LEFT / (WIDTH_LEFT + WIDTH_RIGHT)
    ICHL = IC0 * WIDTH_RATIO
    ICHR = IC0 * (1 - WIDTH_RATIO)

    IC_RATIO = ICHL / ICHR

    # Import data
    FILE_PATH = "/home/omedeiro/"
    FILE_NAME = (
        "SPG806_nMem_ICE_writeEnable_sweep_square11_D6_D1_2023-12-13 02-06-48.mat"
    )
    data_dict = import_matlab_data(FILE_PATH + "/" + FILE_NAME)

    # Extract write enable sweep data.
    enable_write_currents = np.transpose(data_dict["x"][:, :, 1]) * 1e6
    write_currents = np.transpose(data_dict["y"][:, :, 1]) * 1e6
    ber = data_dict["ber"]
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
    N = 50
    enable_write_currents = np.linspace(
        enable_write_currents[0], enable_write_currents[-1], N
    )
    write_currents = np.linspace(write_currents[0], write_currents[-1], N)

    channel_critical_current_enabled = htron_critical_current(
        HTRON_SLOPE, HTRON_INTERCEPT, enable_write_currents
    )

    left_critical_currents = channel_critical_current_enabled * IC_RATIO
    right_critical_currents = channel_critical_current_enabled * (1 - IC_RATIO)

    fig, ax = plt.subplots()
    ax, total_persistent_current, regions = plot_persistent_current(
        ax,
        left_critical_currents,
        write_currents,
        ALPHA,
        ICHL,
        ICHR,
        IRETRAP,
        WIDTH_LEFT,
        WIDTH_RIGHT,
        plot_regions=False,
    )

    ax = plot_point(
        ax,
        left_critical_currents[IDXX],
        write_currents[IDXY],
        marker="*",
        color="red",
        markersize=15,
    )
    # plot_edge_fits(ax, EDGE_FITS, left_critical_currents)

    #     fig, ax = plt.subplots()
    #     ax, read_currents = plot_read_current(
    #         ax,
    #         left_critical_currents,
    #         write_currents,
    #         total_persistent_current,
    #         ALPHA,
    #         ICHL,
    #         ICHR,
    #         IRETRAP,
    #         plot_region=False,
    #     )

    #     ax = plot_point(
    #         ax,
    #         left_critical_currents[IDXX],
    #         write_currents[IDXY],
    #         marker="*",
    #         color="red",
    #         markersize=15,
    #     )

    #     plt.show()

    param_df = get_point_parameters(
        IDXX,
        IDXY,
        left_critical_currents,
        write_currents,
        total_persistent_current,
        enable_write_currents,
        ALPHA,
        IRETRAP,
    )
    print(param_df)

#     # %%
#     fig, ax = plt.subplots()
#     read_margin = plot_read_margin(
#         ax,
#         left_critical_currents,
#         write_currents,
#         read_currents,
#         total_persistent_current,
#         ALPHA,
#         IRETRAP,
#         IREAD,
#     )
#     ax = plot_point(
#         ax,
#         left_critical_currents[IDXX],
#         write_currents[IDXY],
#         marker="*",
#         color="red",
#         markersize=15,
#     )
