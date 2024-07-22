import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from nmem.calculations.calculations import (
    calculate_0_current,
    calculate_1_current,
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


def get_point_parameters(
    i: int,
    j: int,
    critical_currents: np.ndarray,
    write_currents: np.ndarray,
    persistent_currents: np.ndarray,
    enable_currents: np.ndarray,
):
    critical_current = float(critical_currents[i])
    write_current = float(write_currents[j])
    persistent_current = persistent_currents[j, i]
    enable_current = float(enable_currents[i])

    left_branch_write_current = calculate_left_branch_current(ALPHA, write_current, 0)
    right_branch_write_current = calculate_right_branch_current(ALPHA, write_current, 0)

    left_branch_current1 = calculate_left_branch_current(
        ALPHA, write_current, persistent_current
    )
    right_branch_current1 = calculate_right_branch_current(
        ALPHA, write_current, persistent_current
    )

    state_0_current = calculate_0_current(
        critical_current,
        critical_current / ICHL * ICHR,
        ALPHA,
        persistent_current,
        IRETRAP,
    )
    state_1_current = calculate_1_current(
        critical_current,
        critical_current / ICHL * ICHR,
        ALPHA,
        persistent_current,
        IRETRAP,
    )

    ideal_read_current = np.mean([state_0_current, state_1_current])
    ideal_read_margin = np.abs(state_0_current - state_1_current)

    param_dict = {
        "Total Critical Current (enable off)": f"{IC0*1e6:.2f}",
        "Left Critical Current (enable off)": f"{ICHL*1e6:.2f}",
        "Right Critical Current (enable off)": f"{ICHR*1e6:.2f}",
        "Width Ratio": f"{WIDTH_RATIO:.2f}",
        "Inductive Ratio": f"{ALPHA:.2f}",
        "Switching Current Ratio": f"{IC_RATIO:.2f}",
        "Write Current": f"{write_current:.2f}",
        "Enable Write Current": f"{enable_current:.2f}",
        "Left Branch Write Current": f"{left_branch_write_current:.2f}",
        "Right Branch Write Current": f"{right_branch_write_current:.2f}",
        "Left Side Critical Current (enable on)": f"{critical_current:.2f}",
        "Right Side Critical Current (enable on)": f"{critical_current / IC_RATIO:.2f}",
        "Persistent Current": f"{persistent_current:.2f}",
        "State 0 Current": f"{state_0_current:.2f}",
        "State 1 Current": f"{state_1_current:.2f}",
        "Ideal Read Current": f"{ideal_read_current:.2f}",
        "Ideal Read Margin": f"{ideal_read_margin:.2f}",
    }
    param_df = pd.DataFrame(param_dict.values(), index=param_dict.keys())
    param_df.columns = ["Value"]

    return param_df


if __name__ == "__main__":
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.4
    IC0 = 600e-6
    HTRON_SLOPE = -2.69
    HTRON_INTERCEPT = 1000
    ALPHA = 0.628
    BETA = 0.159
    IRETRAP = 0.9

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

    enable_write_currents = np.linspace(
        enable_write_currents[0], enable_write_currents[-1], 50
    )
    write_currents = np.linspace(write_currents[0], write_currents[-1], 50)

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
        ICHR,
        ICHL,
        IRETRAP,
        WIDTH_LEFT,
        WIDTH_RIGHT,
        plot_regions=False,
    )

    # plot_edge_fits(ax, EDGE_FITS, left_critical_currents)

    fig, ax = plt.subplots()
    ax, read_currents = plot_read_current(
        ax,
        left_critical_currents,
        write_currents,
        total_persistent_current,
        ALPHA,
        ICHR,
        ICHL,
        IRETRAP,
        plot_region=True,
    )

    IDXX = 20
    IDXY = 10
    ax = plot_point(
        ax,
        left_critical_currents[IDXX],
        write_currents[IDXY],
        marker="*",
        color="red",
        markersize=15,
    )

    plt.show()

    param_df = get_point_parameters(
        IDXX,
        IDXY,
        left_critical_currents,
        write_currents,
        total_persistent_current,
        enable_write_currents,
    )
    print(param_df)
