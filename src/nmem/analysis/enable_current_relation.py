import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from nmem.analysis.analysis import find_enable_relation, plot_fit

plt.rcParams["figure.figsize"] = [5.7, 5]
plt.rcParams["font.size"] = 16


if __name__ == "__main__":
    files = os.listdir("data")
    markers = ["o", "s", "D", "^"]
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    # colors = np.flipud(colors)
    for file in files:
        datafile = os.path.join("data", file)
        cell = file.split("_")[-2]
        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1

        measurement_dict = sio.loadmat(datafile)
        measurement_dict["cell"] = cell
        xfit, yfit = find_enable_relation(measurement_dict)
        plot_fit(xfit, yfit)
        xfit_sorted = np.sort(xfit)
        yfit_sorted = yfit[np.argsort(xfit)]
        print(f"Cell: {cell} max Ic = {yfit_sorted[0]}")
        print(f"Fit resutls: {np.polyfit(xfit, yfit, 1)}")
        plt.plot(
            xfit_sorted,
            yfit_sorted,
            label=f"Cell {cell}",
            marker=markers[row],
            color=colors[column],
        )

    plt.xlabel("Enable Current [$\mu$A]")
    plt.ylabel("Critical Current [$\mu$A]")
    ax = plt.gca()
    ax.set_ylim(bottom=0)
    # plt.title("Enable Current Relation")
    plt.legend(loc="lower left", ncol=2, frameon=False, fontsize=12)
    plt.show()
