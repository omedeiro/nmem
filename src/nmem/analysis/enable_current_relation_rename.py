import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

plt.rcParams["figure.figsize"] = [5.5, 5.5]

from nmem.analysis.analysis import (
    plot_enable_current_relation,
    plot_slice,
    find_enable_relation,
)


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
