import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os


def find_peak(data_dict: dict):
    x = data_dict["x"][0][:, 1] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6

    w0r1 = 100 - data_dict["write_0_read_1"][0].flatten()
    w1r0 = data_dict["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal, axis=0)

    xfit, yfit = find_enable_relation(data_dict)
    plt.scatter(xfit, yfit)

    # Plot a fit line to the scatter points
    plot_fit(xfit, yfit)

    plt.imshow(
        diff,
        extent=[
            (-0.5 * dx + x[0]),
            (0.5 * dx + x[-1]),
            (-0.5 * dy + y[0]),
            (0.5 * dy + y[-1]),
        ],
        aspect="auto",
        origin="lower",
    )
    plt.xticks(np.linspace(x[0], x[-1], len(x)))
    plt.yticks(np.linspace(y[0], y[-1], len(y)))
    plt.xlabel("Enable Current [$\mu$A]")
    plt.ylabel("Channel Current [$\mu$A]")
    plt.title(f"Cell {data_dict['cell']}")
    cbar = plt.colorbar()
    return ztotal


def find_max_critical_current(data):
    x = data["x"][0][:, 1] * 1e6
    y = data["y"][0][:, 0] * 1e6
    w0r1 = 100 - data["write_0_read_1"][0].flatten()
    w1r0 = data["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")
    ztotal = ztotal[:, 1]

    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal)
    mid_idx = np.where(diff == np.max(diff))

    return np.mean(y[mid_idx])


def find_enable_relation(data_dict: dict):
    x = data_dict["x"][0][:, 1] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6

    w0r1 = 100 - data_dict["write_0_read_1"][0].flatten()
    w1r0 = data_dict["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal, axis=0)
    diff_fit = np.where(diff == 0, np.nan, diff)
    mid_idx = np.where(diff_fit == np.nanmax(diff_fit, axis=0))

    return x[mid_idx[1]], y[mid_idx[0]]


def plot_fit(xfit, yfit):
    z = np.polyfit(xfit, yfit, 1)
    p = np.poly1d(z)
    plt.scatter(xfit, yfit)
    xplot = np.linspace(190, 310, 10)
    plt.plot(xplot, p(xplot), "r--")
    plt.text(
        200,
        350,
        f"{p[1]:.3f}x + {p[0]:.3f}",
        fontsize=12,
        color="red",
        backgroundcolor="white",
    )


if __name__ == "__main__":
    files = os.listdir("data")
    for file in files:
        datafile = os.path.join("data", file)
        cell = file.split("_")[-2]
        measurement_dict = sio.loadmat(datafile)
        measurement_dict["cell"] = cell
        xfit, yfit = find_enable_relation(measurement_dict)
        plot_fit(xfit, yfit)
        find_peak(measurement_dict)
        plt.show()
