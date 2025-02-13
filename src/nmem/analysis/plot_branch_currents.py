import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from nmem.analysis.analysis import (
    plot_branch_currents,
    get_critical_current_intercept,
    import_directory,
)

CRITICAL_TEMP = 12.3
ALPHA = 0.563
RETRAP = 0.573
WIDTH = 1 / 2.13

if __name__ == "__main__":
    data_dict = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\read_current_sweep_enable_read\data"
    )[0]
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
    ax.set_ylim(0, 1500)
    plt.show()
