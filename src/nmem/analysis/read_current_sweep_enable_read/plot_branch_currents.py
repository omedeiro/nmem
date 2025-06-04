import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.core_analysis import (
    get_critical_current_intercept,
)
from nmem.analysis.data_import import import_directory
from nmem.analysis.plotting import plot_branch_currents
CRITICAL_TEMP = 12.3
ALPHA = 0.6
RETRAP = 0.7
WIDTH = 1 / 3

if __name__ == "__main__":
    data_dict = import_directory("data")[0]
    fig, ax = plt.subplots()

    critical_current_zero = get_critical_current_intercept(data_dict) * 0.88
    temps = np.linspace(0, CRITICAL_TEMP, 100)

    plot_branch_currents(ax, temps, CRITICAL_TEMP, RETRAP, WIDTH, critical_current_zero)

    ax.plot()

    ax.axvline(1.3, color="black", linestyle="--", label="Substrate Temp")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [$\mu$A]")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    ax.set_xlim(0, 12.3)
    ax.set_ylim(0, 2000)
    ax.plot([0], [critical_current_zero], marker="x", color="black", markersize=10)
    ax.plot([7], [800], marker="x", color="black", markersize=10)
    save_fig = False
    if save_fig:
        plt.savefig("plot_branch_currents.pdf", bbox_inches="tight")
    plt.show()
