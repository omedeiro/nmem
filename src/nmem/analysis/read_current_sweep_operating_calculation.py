import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from nmem.analysis.analysis import (
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_read_current,
    get_read_currents,
    get_state_current_markers,
    get_write_current,
    import_directory,
    plot_read_sweep_array,
    plot_read_switch_probability_array,
    calculate_critical_current_temp,
    calculate_channel_temperature,
    calculate_state_currents,
    CELLS,
    RBCOLORS,
)

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


CRITICAL_CURRENT_ZERO = 1250
WIDTH = 0.3



def calculate_inductance_ratio(state0, state1, ic0):
    alpha = (ic0 - state1) / (state0 - state1)
    # alpha_test = 1 - ((critical_current_right - persistent_current_est) / ic)
    # alpha_test2 = (critical_current_left - persistent_current_est) / ic2

    return alpha


if __name__ == "__main__":
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\read_current_sweep_write_current2\write_current_sweep_C3"
    )

    ic_list = []
    write_current_list = []
    ic_list2 = []
    write_current_list2 = []
    for data_dict in dict_list:
        write_current = get_write_current(data_dict)

        bit_error_rate = get_bit_error_rate(data_dict)
        
        berargs = get_bit_error_rate_args(bit_error_rate)
        read_currents = get_read_currents(data_dict)
        if not np.isnan(berargs[0]) and write_current < 100:
            ic_list.append(read_currents[berargs[0]])
            write_current_list.append(write_current)
        if not np.isnan(berargs[2]) and write_current > 100:
            ic_list.append(read_currents[berargs[3]])
            write_current_list.append(write_current)

        if not np.isnan(berargs[1]):
            ic_list2.append(read_currents[berargs[1]])
            write_current_list2.append(write_current)
        if not np.isnan(berargs[3]):
            ic_list2.append(read_currents[berargs[2]])
            write_current_list2.append(write_current)

    ic = np.array(ic_list)
    ic2 = np.array(ic_list2)
    write_current_array = np.array(write_current_list)
    read_temperature = calculate_channel_temperature(
        CRITICAL_TEMP,
        SUBSTRATE_TEMP,
        data_dict["enable_read_current"] * 1e6,
        CELLS[data_dict["cell"][0]]["x_intercept"],
    ).flatten()
    write_temperature = calculate_channel_temperature(
        CRITICAL_TEMP,
        SUBSTRATE_TEMP,
        data_dict["enable_write_current"] * 1e6,
        CELLS[data_dict["cell"][0]]["x_intercept"],
    ).flatten()


    critical_current_channel = calculate_critical_current_temp(
        read_temperature, CRITICAL_TEMP, CRITICAL_CURRENT_ZERO
    )
    critical_current_left = critical_current_channel * WIDTH
    critical_current_right = critical_current_channel * (1 - WIDTH)


    pd = pd.DataFrame(
        {
            "Write Current": write_current_list,
            "Read Current": ic_list,
            "Read Current 2": ic_list2,
            "Channel Critical Current": critical_current_channel*np.ones_like(ic_list),
            "Channel Critical Current Left": critical_current_left*np.ones_like(ic_list),
            "Channel Critical Current Right": critical_current_right*np.ones_like(ic_list),
        }
    )
    
    pd.to_csv("read_current_sweep_operating.csv")

