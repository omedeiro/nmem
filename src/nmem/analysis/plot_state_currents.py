import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import plot_calculated_state_currents

if __name__ == "__main__":
    fig, ax = plt.subplots()

    critical_current_zero = 1
    plot_calculated_state_currents(ax, np.linspace(0,12.3, 100), 12.3, 0.5, 0.3, 0.6, 0, critical_current_zero)

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [au]")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid()
    ax.set_xlim(0, 12.3)
    # ax.set_ylim(0, 1)
    plt.show()