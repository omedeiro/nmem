from nmem.analysis.analysis import plot_branch_currents
import matplotlib.pyplot as plt
import numpy as np

CRITICAL_TEMP = 12.3
CRITICAL_CURRENT_ZERO = 1300
ALPHA = 0.563
RETRAP = 0.573
WIDTH = 1 / 3

if __name__ == "__main__":
    fig, ax = plt.subplots()
    temps = np.linspace(0, CRITICAL_TEMP, 100)
    plot_branch_currents(
        ax, temps, CRITICAL_TEMP, RETRAP, WIDTH, CRITICAL_CURRENT_ZERO
    )
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [au]")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    ax.set_xlim(0, 12.3)
    plt.show()
