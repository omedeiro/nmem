import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def htron_critical_current(slope, intercept, heater_current):
    return heater_current * slope + intercept


def import_matlab_data(file_path):
    data = sio.loadmat(file_path)
    return data


def calculate_right_branch_inductance(alpha, ll):
    return alpha * ll / (1 - alpha)


def calculate_left_branch_inductance(alpha, lr):
    return (1 - alpha) * lr / alpha


def calculate_right_branch_current(alpha, channel_current, persistent_current):
    """State 1 is defined as a positive persistent current, clockwise
    State 0 is defined as a negative persistent current, counter-clockwise
    Right branch sums current during read.
    """
    return channel_current * (1 - alpha) + persistent_current


def calculate_left_branch_current(alpha, channel_current, persistent_current):
    """State 1 is defined as a positive persistent current, clockwise
    State 0 is defined as a negative persistent current, counter-clockwise
    Left branch negates current during read.
    """
    return channel_current * alpha - persistent_current


def caluclate_branch_critical_currents(critical_current, width_left, width_right):
    ratio = width_left / (width_left + width_right)
    return critical_current * np.array([ratio, 1 - ratio])


def calculate_0_current(ichl, ichr, alpha, persistent_current):
    return np.max([(ichl - persistent_current) / alpha, ichl * IRETRAP + ichr])


def calculate_1_current(ichl, ichr, alpha, persistent_current):
    return np.max([(ichr - persistent_current) / (1 - alpha), ichr * IRETRAP + ichl])


def calculate_alpha(ll, lr):
    """ll < lr"""
    return lr / (ll + lr)


def calculate_persistent_current(critical_current, write_current, alpha):
    persistent_current = write_current * alpha - critical_current

    # Exclude persistent current values that are greater than the critical current
    persistent_current[np.abs(persistent_current) > np.abs(critical_current)] = 0

    return persistent_current


def calculate_inverting_persistent_current(critical_current, write_current, alpha):
    persistent_current = write_current * alpha - critical_current

    # Exclude persistent current values that are greater than the critical current
    persistent_current[np.abs(persistent_current) > np.abs(critical_current)] = 0

    persistent_current[write_current < critical_current / alpha - 226] = 0
    persistent_current[write_current > critical_current / alpha - 165] = 0

    return persistent_current


def calculate_non_inverting_persistent_current(critical_current, write_current, beta):
    persistent_current = -write_current * beta + critical_current

    persistent_current[np.abs(persistent_current) > np.abs(critical_current)] = 0

    persistent_current[write_current < critical_current / beta - 433] = 0
    persistent_current[write_current > critical_current / beta - 353] = 0

    return persistent_current


def calculate_total_persistent_current(critical_current, write_current, alpha, beta):
    total_persistent_current = calculate_inverting_persistent_current(
        critical_current, write_current, alpha
    ) + calculate_non_inverting_persistent_current(
        critical_current, write_current, beta
    )
    total_persistent_current = np.where(
        total_persistent_current > critical_current, 0, total_persistent_current
    )
    return total_persistent_current


def calculate_read_currents(critical_currents, write_currents, persistent_currents):
    [xx, yy] = np.meshgrid(critical_currents, write_currents)
    read_currents = np.zeros((len(write_currents), len(critical_currents)))
    for i in range(len(write_currents)):
        for j in range(len(critical_currents)):

            ichr = xx[i, j] * ICHR / ICHL
            if persistent_currents[i, j] > 0:  # Memory "1"
                read_currents[i, j] = (ichr - persistent_currents[i, j]) / (1 - ALPHA)
            elif persistent_currents[i, j] < 0:  # Memory "0"
                read_currents[i, j] = (ichr + persistent_currents[i, j]) / (1 - BETA)
            else:
                read_currents[i, j] = 0

            read_currents[i, j] = np.abs(read_currents[i, j])
            # Read current cannot be less than the write current
            # if read_currents[i, j] < yy[i, j]:
            #     read_currents[i, j] = 0

            # Negative read currents are not possible
            # if read_currents[i, j] < 0:
            #     read_currents[i, j] = 0

    return read_currents


def get_point_parameters(
    i: int, j: int, alpha: float, critical_currents, write_currents, persistent_currents
):
    critical_current = critical_currents[i]
    write_current = write_currents[j]
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
        critical_current, critical_current / ICHL * ICHR, alpha, persistent_current
    )
    state_1_current = calculate_1_current(
        critical_current, critical_current / ICHL * ICHR, alpha, persistent_current
    )

    ideal_read_current = np.mean([state_0_current, state_1_current])
    ideal_read_margin = np.abs(state_0_current - state_1_current)

    params = {
        "Write Current": write_current,
        "Persistent Current": persistent_current,
        "Left Side Critical Current": critical_current,
        "Right Side Critical Current": critical_current / ICHL * ICHR,
        "Write Left Branch Current 0/1 ": [left_branch_current0, left_branch_current1],
        "Write Right Branch Current 0/1 ": [
            right_branch_current0,
            right_branch_current1,
        ],
        "State 0 Current": state_0_current,
        "State 1 Current": state_1_current,
        "Ideal Read Current": ideal_read_current,
        "Ideal Read Margin": ideal_read_margin,
    }
    return params


def plot_point(ax, x, y, **kwargs):
    ax.plot(x, y, **kwargs)
    return ax


def plot_htron_critical_current(slope, intercept, currents):

    x = currents
    y = htron_critical_current(slope, intercept, x)

    plt.plot(x, y)
    plt.xlabel("Enable Current [uA]")
    plt.ylabel("Critical Current [uA]")
    plt.title("Critical Current vs Enable Current")
    plt.show()


def print_dict_keys(data_dict):
    for key in data_dict.keys():
        print(key)


def plot_htron_sweep(write_currents, enable_write_currents, ber, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    xx, yy = np.meshgrid(enable_write_currents, write_currents)
    plt.contourf(xx, yy, ber)
    # plt.gca().invert_xaxis()
    plt.xlabel("Enable Current [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("BER vs Write Current and Critical Current")
    plt.colorbar()
    return ax


def plot_persistent_current(ax, critical_currents, write_currents):
    [xx, yy] = np.meshgrid(critical_currents, write_currents)

    total_persistent_current = calculate_total_persistent_current(xx, yy, ALPHA, BETA)

    plt.pcolormesh(xx, yy, total_persistent_current)
    plt.xlabel("Left Branch Critical Current ($I_{C, H_L}(I_{RE})$)) [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("Persistent Current")
    plt.gca().invert_xaxis()
    plt.colorbar()
    # plt.clim([np.min(persistent_current[persistent_current<0]), np.max(persistent_current)])
    return ax, total_persistent_current


def plot_read_current(ax, critical_currents, write_currents, persistent_currents):
    read_currents = calculate_read_currents(
        critical_currents, write_currents, persistent_currents
    )
    [xx, yy] = np.meshgrid(critical_currents, write_currents)
    plt.pcolormesh(xx, yy, read_currents)
    plt.xlabel("Critical Current [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("Read Current")
    plt.gca().invert_xaxis()
    plt.colorbar()
    return ax


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


def calculate_critical_current_bounds(persistent_current, read_current, alpha):
    return read_current * alpha * np.ones((2, 1)) + [
        -persistent_current,
        persistent_current,
    ]


if __name__ == "__main__":
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.3
    IC0 = 600e-6
    HTRON_SLOPE = -2.69
    HTRON_INTERCEPT = 1257
    ALPHA = 0.356
    BETA = 0.159
    IRETRAP = 0.5
    ICHL, ICHR = caluclate_branch_critical_currents(IC0, WIDTH_LEFT, WIDTH_RIGHT)
    IC_RATIO = ICHL / (ICHL + ICHR)
    print(f"Left critical current: {ICHL*1e6} uA")
    print(f"Right critical current: {ICHR*1e6} uA")

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
    ax = plot_htron_sweep(write_currents, enable_write_currents, ber_2D)

    lines = [
        {"p1": 2.818, "p2": -226.1},
        {"p1": 2.818, "p2": -165.1},
        {"p1": 6.272, "p2": -433.9},
        {"p1": 6.272, "p2": -353.8},
    ]

    left_critical_currents = (
        htron_critical_current(HTRON_SLOPE, HTRON_INTERCEPT, enable_write_currents)
        * IC_RATIO
    )
    fig, ax = plt.subplots()
    ax, total_persistent_current = plot_persistent_current(
        ax, left_critical_currents, write_currents
    )
    plot_edge_fits(ax, lines, left_critical_currents)

    # fig, ax = plt.subplots()
    # ax = plot_read_current(
    #     ax, left_critical_currents, write_currents, total_persistent_current
    # )
    # IDXX = 10
    # IDXY = 10
    # ax = plot_point(
    #     ax,
    #     left_critical_currents[IDXX],
    #     write_currents[IDXY],
    #     marker="*",
    #     color="red",
    #     markersize=15,
    # )

    # plt.show()

    # params = get_point_parameters(
    #     IDXX, IDXY, ALPHA, left_critical_currents, write_currents, total_persistent_current
    # )
    # for key, value in params.items():
    #     print(f"{key}: {value}")
