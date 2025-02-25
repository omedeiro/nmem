from nmem.analysis.analysis import (
    plot_calculated_state_currents,
    plot_calculated_filled_region,
    import_directory,
    get_critical_current_intercept,
    CMAP,
    filter_nan,
)
import matplotlib.pyplot as plt
import numpy as np
from nmem.measurement.functions import calculate_power
import scipy.io as sio
SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


ALPHA = 0.23
RETRAP = 1
WIDTH = 0.3


if __name__ == "__main__":

    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )
    data_dict = dict_list[0]
    power = calculate_power(data_dict)
    persistent_current = 75
    fig, ax = plt.subplots()
    critical_current_zero = 1250

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


    
    data_dict1 = sio.loadmat("measured_state_currents_290.mat")
    data_dict2 = sio.loadmat("measured_state_currents_300.mat")
    data_dict3 = sio.loadmat("measured_state_currents_310.mat")

    dict_list = [data_dict1, data_dict2, data_dict3]
    colors = {0: "blue", 1: "blue", 2: "red", 3: "red"}
    fit_results = []
    for data_dict in [dict_list[1]]:
        temp = data_dict["measured_temperature"].flatten()
        state_currents = data_dict["measured_state_currents"]
        x_list = []
        y_list = []
        for i in range(4):
            x = temp
            y = state_currents[:, i]
            x, y = filter_nan(x, y)
            ax.plot(x, y, "-o", color=colors[i], label=f"State {i}")


            
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Current [au]")
    ax.grid()
    ax.set_xlim(0, CRITICAL_TEMP)
    ax.plot([7], [800], marker="x", color="black", markersize=10)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlim(6, 9)
    ax.set_ylim(500, 900)
    # ax.set_ylim(0, 1)
    plt.show()
