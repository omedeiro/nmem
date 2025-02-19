from nmem.analysis.analysis import (
    plot_calculated_state_currents,
    plot_calculated_filled_region,
    import_directory,
    get_critical_current_intercept,
)
import matplotlib.pyplot as plt
import numpy as np
from nmem.measurement.functions import calculate_power

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


ALPHA = 0.4
RETRAP = 0.7
WIDTH = 0.33

if __name__ == "__main__":

    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )
    data_dict = dict_list[0]
    power = calculate_power(data_dict)
    persistent_current = 30
    fig, ax = plt.subplots()
    critical_current_zero = get_critical_current_intercept(data_dict)*0.88

    temperatures = np.linspace(0, CRITICAL_TEMP, 100)
    
    plot_calculated_state_currents(
        ax,
        temperatures,
        CRITICAL_TEMP,
        RETRAP,
        WIDTH,
        ALPHA,
        persistent_current,
        critical_current_zero,
    )

    plot_calculated_filled_region(
        ax,
        temperatures,
        data_dict,
        persistent_current,
        CRITICAL_TEMP,
        RETRAP,
        WIDTH,
        ALPHA,
        critical_current_zero,
    )
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [au]")
    ax.grid()
    ax.set_xlim(0, CRITICAL_TEMP)
    ax.plot([7], [800], marker="x", color="black", markersize=10)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

    # ax.set_ylim(0, 1)
    plt.show()
