import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams["figure.figsize"] = [5.5, 5.5]


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
    # xfit = xfit[:-8]
    # yfit = yfit[:-8]

    plt.scatter(xfit, yfit)

    # Plot a fit line to the scatter points
    plot_fit(xfit, yfit)

    plt.imshow(
        ztotal,
        extent=[
            (-0.5 * dx + x[0]),
            (0.5 * dx + x[-1]),
            (-0.5 * dy + y[0]),
            (0.5 * dy + y[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    plt.xticks(np.linspace(x[0], x[-1], len(x)))
    plt.yticks(np.linspace(y[0], y[-1], len(y)))
    plt.xlabel("Enable Current [$\mu$A]")
    plt.ylabel("Channel Current [$\mu$A]")
    plt.title(f"Cell {data_dict['cell']}")
    cbar = plt.colorbar()
    cbar.set_label("Counts")
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
    # diff = np.diff(ztotal, axis=0)
    # diff_fit = np.where(diff == 0, np.nan, diff)
    # mid_idx = np.where(diff_fit == np.nanmax(diff_fit, axis=0))

    # Find the maximum critical current using total counts
    mid_idx = np.where(ztotal >= np.nanmax(ztotal - 5, axis=0))
    xfit, xfit_idx = np.unique(x[mid_idx[1]], return_index=True)
    yfit = y[mid_idx[0]][xfit_idx]
    return xfit, yfit


def plot_fit(xfit, yfit):
    z = np.polyfit(xfit, yfit, 1)
    p = np.poly1d(z)
    plt.scatter(xfit, yfit, color="#08519C")
    xplot = np.linspace(190, 310, 10)
    plt.plot(xplot, p(xplot), "--", color="#740F15")
    # plt.text(
    #     200,
    #     350,
    #     f"{p[1]:.3f}x + {p[0]:.3f}",
    #     fontsize=12,
    #     color="red",
    #     backgroundcolor="white",
    # )


def plot_enable_current_relation(data_dict: dict):
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
    # xfit = xfit[:-8]
    # yfit = yfit[:-8]

    # plt.scatter(xfit, yfit, color="#740F15")

    # Plot a fit line to the scatter points
    plot_fit(xfit, yfit)

    plt.imshow(
        ztotal,
        extent=[
            (-0.5 * dx + x[0]),
            (0.5 * dx + x[-1]),
            (-0.5 * dy + y[0]),
            (0.5 * dy + y[-1]),
        ],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    # plt.xticks(np.linspace(200, 300, 3))
    # plt.yticks(np.linspace(300, 900, 3))
    plt.xlim(210, 280)
    plt.ylim(200, 500)
    plt.xlabel("Enable Current [$\mu$A]")
    plt.ylabel("Channel Current [$\mu$A]")
    plt.title("Enable Current Relation")
    cbar = plt.colorbar()
    cbar.set_label("Counts")
    plt.show()

    return ztotal


def plot_slice(data_dict: dict, index: int):
    x = data_dict["x"][0][:, 1] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6

    w0r1 = 100 - data_dict["write_0_read_1"][0].flatten()
    w1r0 = data_dict["write_1_read_0"][0].flatten()
    z = w1r0 + w0r1
    ztotal = z.reshape((len(y), len(x)), order="F")

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    zslice = ztotal[:, index]
    plt.plot(y, zslice, marker=".", linewidth=2, color="#08519C")
    plt.xlabel("Channel Current [$\mu$A]")
    plt.ylabel("Counts")
    plt.title(f"Enable Current Slice at {x[index]:.2f} $\mu$A")
    plt.show()

    return zslice


if __name__ == "__main__":
    files = os.listdir("data2")
    markers = ["o", "s", "D", "^"]
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    # colors = np.flipud(colors)
    for file in [files[0]]:
        datafile = os.path.join("data2", file)
        cell = file.split("_")[-2]
        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1

        measurement_dict = sio.loadmat(datafile)
        measurement_dict["cell"] = cell
        xfit, yfit = find_enable_relation(measurement_dict)
        xfit_sorted = np.sort(xfit)
        yfit_sorted = yfit[np.argsort(xfit)]
        print(f"Cell: {cell} max Ic = {yfit_sorted[0]}")

        plot_enable_current_relation(measurement_dict)
        # plt.show()
        # # plot_fit(xfit_sorted, yfit_sorted)
        # plt.plot(xfit_sorted, yfit_sorted, label=f"Cell {cell}", marker=markers[row], color=colors[column])
        # w0r1 = 100 - measurement_dict["write_0_read_1"][0].flatten()
        # w1r0 = measurement_dict["write_1_read_0"][0].flatten()
        # z = w1r0 + w0r1
        # ztotal = z.reshape(
        #     (measurement_dict["y"][0].shape[0], measurement_dict["x"][0].shape[0]),
        #     order="F",
        # )
        # cmap = plt.cm.viridis(np.linspace(0, 1, ztotal.shape[1]))

    plt.xlabel("Enable Current [$\mu$A]")
    plt.ylabel("Critical Current [$\mu$A]")
    ax = plt.gca()
    ax.set_ylim(bottom=0)
    plt.title(f"Cell {cell}")
    plt.legend()
    plt.show()

    y = plot_slice(measurement_dict, 1)
