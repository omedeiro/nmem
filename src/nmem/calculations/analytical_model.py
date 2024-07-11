import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


def htron_critical_current(slope, intercept, current):
    return (slope * current + intercept) / 4


def import_matlab_data(file_path):
    data = sio.loadmat(file_path)
    return data


def calculate_right_branch_inductance(alpha, ll):
    return alpha * ll / (1 - alpha)


def calculate_left_branch_inductance(alpha, lr):
    return (1 - alpha) * lr / alpha


def calculate_alpha(ll, lr):
    '''  ll < lr '''
    return lr / (ll + lr)


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
    plt.xlabel("Critical Current [uA]")

    plt.gca().invert_xaxis()

    # # Plot a top x axis
    # top_x_axis = plt.twiny()
    # top_x_axis.set_xlim(critical_currents[0], critical_currents[-1])
    # top_x_axis.set_xticks(critical_currents)
    # top_x_axis.set_xticklabels(f"{critical_currents:.0f}" for critical_currents in critical_currents)
    # top_x_axis.set_xlabel("Enable Current [uA]")

    plt.ylabel("Write Current [uA]")
    plt.title("BER vs Write Current and Critical Current")
    return ax

def plot_edge_fits(ax, lines, critical_currents):
    for l in lines:
        plot_edge_fit(ax, critical_currents, **l)

def plot_edge_fit(ax, x, p1, p2):
    y = p1 * x + p2
    ax.plot(x, y, color="red")


if __name__ == "__main__":
    HTRON_SLOPE = -2.69
    HTRON_INTERCEPT = 1257
    ALPHA = 0.356
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

    plot_edge_fits(ax, lines, critical_currents)
    # print(import_matlab_data(file_path))
