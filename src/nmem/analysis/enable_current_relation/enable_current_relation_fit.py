import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from nmem.analysis.analysis import build_array, get_fitting_points, plot_fit

plt.rcParams["figure.figsize"] = [5.7, 5]
plt.rcParams["font.size"] = 16


if __name__ == "__main__":
    files = os.listdir("data")
    markers = ["o", "s", "D", "^"]
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    # colors = np.flipud(colors)
    fig, ax = plt.subplots()

    for file in files:
        datafile = os.path.join("data", file)
        cell = file.split("_")[-2]
        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1

        measurement_dict = sio.loadmat(datafile)
        measurement_dict["cell"] = cell
        x, y, ztotal = build_array(measurement_dict, "bit_error_rate")
        xfit, yfit = get_fitting_points(x, y, ztotal)
        plot_fit(ax, xfit, yfit)

        xfit_sorted = np.sort(xfit)
        yfit_sorted = yfit[np.argsort(xfit)]
        ax.plot(
            xfit_sorted,
            yfit_sorted,
            label=f"Cell {cell}",
            marker=markers[row],
            color=colors[column],
        )

    ax.set_xlabel("Enable Current [$\mu$A]")
    ax.set_ylabel("Critical Current [$\mu$A]")
    ax.set_ylim(bottom=0)

    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=12)
