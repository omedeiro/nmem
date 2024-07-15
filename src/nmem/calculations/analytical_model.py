import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def htron_critical_current(slope, intercept, current):
    return (slope * current + intercept) / 4


def import_matlab_data(file_path):
    data = sio.loadmat(file_path)
    return data


def calculate_right_branch_inductance(alpha, ll):
    return alpha * ll / (1 - alpha)


def calculate_left_branch_inductance(alpha, lr):
    return (1 - alpha) * lr / alpha


def caluclate_branch_critical_currents(critical_current, width_left, width_right):
    ratio = width_left / (width_left + width_right)
    return critical_current * np.array([ratio, 1 - ratio])


def calculate_alpha(ll, lr):
    """ll < lr"""
    return lr / (ll + lr)


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
    return total_persistent_current


def calculate_read_currents(critical_currents, write_currents, persistent_currents):
    [xx, yy] = np.meshgrid(critical_currents, write_currents)
    read_currents = np.zeros((len(write_currents), len(critical_currents)))
    for i in range(len(write_currents)):
        for j in range(len(critical_currents)):
            if persistent_currents[i, j] > 0: # Memory "1"
                read_currents[i, j] = (xx[i,j] * 3/4 - persistent_currents[i, j])/(1-ALPHA)
            elif persistent_currents[i, j] < 0: # Memory "0"
                read_currents[i, j] = (xx[i,j] * 3/4 + persistent_currents[i, j])/(1-BETA)
            else:
                read_currents[i, j] = 0
            # if read_currents[i, j] < yy[i, j]: # Read current is greater than write current. 
            #     read_currents[i, j] = 0
    return read_currents


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
    critical_currents = htron_critical_current(
        HTRON_SLOPE, HTRON_INTERCEPT, enable_write_currents
    ).flatten()

    xx, yy = np.meshgrid(critical_currents, write_currents)
    plt.contourf(xx, yy, ber)
    plt.gca().invert_xaxis()
    plt.xlabel("Critical Current [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("BER vs Write Current and Critical Current")
    return ax


def plot_persistent_current(ax, critical_currents, write_currents):
    [xx, yy] = np.meshgrid(critical_currents, write_currents)

    total_persistent_current = calculate_total_persistent_current(xx, yy, ALPHA, BETA)
    plt.pcolormesh(xx, yy, total_persistent_current)
    plt.xlabel("Critical Current [uA]")
    plt.ylabel("Write Current [uA]")
    plt.title("Inverting Persistent Current")
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


def plot_edge_fit(ax, x, p1, p2):
    y = p1 * x + p2
    ax.plot(x, y, color="red")


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
    ICHL, ICHR = caluclate_branch_critical_currents(IC0, WIDTH_LEFT, WIDTH_RIGHT)
    print(f"Left critical current: {ICHL*1e6} uA")
    print(f"Right critical current: {ICHR*1e6} uA")

    FILE_PATH = "/home/omedeiro/"
    FILE_NAME = (
        "SPG806_nMem_ICE_writeEnable_sweep_square11_D6_D1_2023-12-13 02-06-48.mat"
    )

    data_dict = import_matlab_data(FILE_PATH + "/" + FILE_NAME)

    enable_write_currents = np.transpose(data_dict["x"][:, :, 1]) * 1e6
    write_currents = np.transpose(data_dict["y"][:, :, 1]) * 1e6
    ber = data_dict["ber"]
    ber_2D = np.reshape(ber, (len(write_currents), len(enable_write_currents)))
    ax = plot_htron_sweep(write_currents, enable_write_currents, ber_2D)
    critical_currents = htron_critical_current(
        HTRON_SLOPE, HTRON_INTERCEPT, enable_write_currents
    )
    lines = [
        {"p1": 2.818, "p2": -226.1},
        {"p1": 2.818, "p2": -165.1},
        {"p1": 6.272, "p2": -433.9},
        {"p1": 6.272, "p2": -353.8},
    ]

    fig, ax = plt.subplots()
    ax, total_persistent_current = plot_persistent_current(
        ax, critical_currents, write_currents
    )
    plot_edge_fits(ax, lines, critical_currents)

    fig, ax = plt.subplots()
    ax = plot_read_current(
        ax, critical_currents, write_currents, total_persistent_current
    )
