import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from nmem.calculations.calculations import (
    calculate_0_current,
    calculate_1_current,
    calculate_left_branch_current,
    calculate_right_branch_current,
    caluclate_branch_critical_currents,
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
    alpha: float,
    critical_currents: np.ndarray,
    write_currents: np.ndarray,
    persistent_currents: np.ndarray,
    iretrap: float,
):
    critical_current = float(critical_currents[i])
    write_current = float(write_currents[j])
    persistent_current = persistent_currents[i, j]
    left_branch_current0 = calculate_left_branch_current(
        alpha, write_current, -persistent_current
    )
    right_branch_current0 = calculate_right_branch_current(
        alpha, write_current, -persistent_current
    )

    left_branch_current1 = calculate_left_branch_current(
        alpha, write_current, persistent_current
    )
    right_branch_current1 = calculate_right_branch_current(
        alpha, write_current, persistent_current
    )

    state_0_current = calculate_0_current(
        critical_current,
        critical_current / ICHL * ICHR,
        alpha,
        persistent_current,
        iretrap,
    )
    state_1_current = calculate_1_current(
        critical_current,
        critical_current / ICHL * ICHR,
        alpha,
        persistent_current,
        iretrap,
    )

    ideal_read_current = np.mean([state_0_current, state_1_current])
    ideal_read_margin = np.abs(state_0_current - state_1_current)

    params = {
        "Write Current": f"{write_current:.2f}",
        "Persistent Current": f"{persistent_current:.2f}",
        "Left Side Critical Current": f"{critical_current:.2f}",
        "Right Side Critical Current": f"{critical_current * ICHR / ICHL:.2f}",
        "Write Left Branch Current 0/1 ": f"{left_branch_current0:.2f}/{left_branch_current1:.2f}",
        "Write Right Branch Current 0/1 ": f"{right_branch_current0:.2f}/{right_branch_current1:.2f}",
        "State 0 Current": f"{state_0_current:.2f}",
        "State 1 Current": f"{state_1_current:.2f}",
        "Ideal Read Current": f"{ideal_read_current:.2f}",
        "Ideal Read Margin": f"{ideal_read_margin:.2f}",
    }
    return params


if __name__ == "__main__":
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.3
    IC0 = 600e-6
    HTRON_SLOPE = -2.69
    HTRON_INTERCEPT = 1057
    ALPHA = -1 / HTRON_SLOPE
    BETA = 0.159
    IRETRAP = 0.9
    ICHL, ICHR = caluclate_branch_critical_currents(IC0, WIDTH_LEFT, WIDTH_RIGHT)
    IC_RATIO = WIDTH_LEFT / WIDTH_RIGHT
    print(f"Left critical current at I_en=0: {ICHL*1e6} uA")
    print(f"Right critical current I_en=0: {ICHR*1e6} uA")

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

    left_critical_currents = (
        htron_critical_current(HTRON_SLOPE, HTRON_INTERCEPT, enable_write_currents)
        * IC_RATIO
    )

    fig, ax = plt.subplots()
    ax, total_persistent_current = plot_persistent_current(
        ax,
        left_critical_currents,
        write_currents,
        ALPHA,
        ICHR,
        ICHL,
        IRETRAP,
        WIDTH_LEFT,
        WIDTH_RIGHT,
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
    )

    IDXX = 10
    IDXY = 35
    ax = plot_point(
        ax,
        left_critical_currents[IDXX],
        write_currents[IDXY],
        marker="*",
        color="red",
        markersize=15,
    )

    plt.show()

    # print(
    #     calculate_left_branch_limits(
    #         ALPHA, read_currents[IDXX, IDXY], total_persistent_current[IDXX, IDXY]
    #     )
    # )

    # print(
    #     calculate_right_branch_limits(
    #         ALPHA, read_currents[IDXX, IDXY], total_persistent_current[IDXX, IDXY]
    #     )
    # )
    # plt.show()

    param_dict = {
        "Critical Current Ratio": f"{IC_RATIO:.2f}",
        "Inductive Ratio": f"{ALPHA:.2f}",
        "Channel Critical Current (uA)": f"{IC0*1e6:.2f}",
    }
    params = get_point_parameters(
        IDXX,
        IDXY,
        ALPHA,
        left_critical_currents,
        write_currents,
        total_persistent_current,
        IRETRAP,
    )
    for key, value in params.items():
        print(f"{key}: {value}")

    for key, value in param_dict.items():
        print(f"{key}: {value}")
