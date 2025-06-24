import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.constants import CRITICAL_TEMP
from nmem.analysis.currents import get_critical_current_intercept
from nmem.analysis.data_import import import_directory
from nmem.analysis.state_currents_plots import plot_branch_currents

ALPHA = 0.6
RETRAP = 0.7
WIDTH = 1 / 3


def main(data_dir="../data/ber_sweep_read_current/nominal", save_dir=None):
    data_dict = import_directory(data_dir)[0]
    fig, ax = plt.subplots()

    critical_current_zero = get_critical_current_intercept(data_dict) * 0.88
    temps = np.linspace(0, CRITICAL_TEMP, 100)

    plot_branch_currents(ax, temps, CRITICAL_TEMP, RETRAP, WIDTH, critical_current_zero)

    ax.plot()

    ax.axvline(1.3, color="black", linestyle="--", label="Substrate Temp")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [$\\mu$A]")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    ax.set_xlim(0, 12.3)
    ax.set_ylim(0, 2000)
    ax.plot([0], [critical_current_zero], marker="x", color="black", markersize=10)
    ax.plot([7], [800], marker="x", color="black", markersize=10)

    if save_dir:
        plt.savefig(
            f"{save_dir}/ber_branch_currents_sweep.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
