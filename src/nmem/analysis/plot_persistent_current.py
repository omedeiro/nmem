import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from nmem.analysis.analysis import (
    ALPHA,
    CRITICAL_TEMP,
    RETRAP,
    WIDTH,
    filter_nan,
    get_critical_current_intercept,
    import_directory,
    plot_calculated_filled_region,
    plot_calculated_state_currents,
)
from nmem.measurement.functions import calculate_power


def plot_measured_markers(ax: plt.Axes, data_dict: dict) -> plt.Axes:
    colors = {0: "blue", 1: "blue", 2: "red", 3: "red"}

    temp = data_dict["measured_temperature"].flatten()
    state_currents = data_dict["measured_state_currents"]

    for i in range(4):
        x = temp
        y = state_currents[:, i]
        x, y = filter_nan(x, y)
        ax.plot(x, y, "-o", color=colors[i], label=f"State {i}")

    return ax



if __name__ == "__main__":
    # Import
    trace_meas_dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )
    trace_meas_dict = trace_meas_dict_list[0]

    data_dict1 = sio.loadmat("measured_state_currents_290.mat")
    data_dict2 = sio.loadmat("measured_state_currents_300.mat")
    data_dict3 = sio.loadmat("measured_state_currents_310.mat")
    dict_list = [data_dict1]

    # Preprocess
    power = calculate_power(trace_meas_dict)
    persistent_current = 0
    critical_current_zero = get_critical_current_intercept(trace_meas_dict)

    temperatures = np.linspace(0, CRITICAL_TEMP, 100)




    # Plot
    fig, ax = plt.subplots()
    for data_dict in dict_list:
        plot_measured_markers(ax, data_dict)


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
        trace_meas_dict,
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
    ax.set_xlim(6, 9)
    ax.set_ylim(500, 900)
    plt.show()
