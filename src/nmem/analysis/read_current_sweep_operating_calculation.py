import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from nmem.analysis.analysis import (
    CELLS,
    RBCOLORS,
    calculate_channel_temperature,
    calculate_critical_current_temp,
    calculate_state_currents,
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_read_current,
    get_read_currents,
    get_state_current_markers,
    get_write_current,
    import_directory,
    plot_read_sweep_array,
    plot_read_switch_probability_array,
)

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3
IRM = 727
IRHL_TR = 110
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
    read_current_difference = ic2 - ic
    read1_dist = np.abs(ic - IRM)
    read2_dist = np.abs(ic2 - IRM)
    persistent_current = []
    for i, write_current in enumerate(write_current_list):
        if write_current > IRHL_TR / 2:
            ip = np.abs(write_current - IRHL_TR) / 2
        else:
            ip = write_current
        if ip > IRHL_TR / 2:
            ip = IRHL_TR / 2
        print(f"write_current: {write_current}, persistent current: {ip}")
        persistent_current.append(ip)
    pd = pd.DataFrame(
        {
            "Write Current": write_current_list,
            "Read Current": ic_list,
            "Read Current 2": ic_list2,
            "Channel Critical Current": critical_current_channel
            * np.ones_like(ic_list),
            "Channel Critical Current Left": critical_current_left
            * np.ones_like(ic_list),
            "Channel Critical Current Right": critical_current_right
            * np.ones_like(ic_list),
            "Persistent Current": persistent_current,
            "Read Current Difference": read_current_difference,
            "Read Current 1 Distance": read1_dist,
            "Read Current 2 Distance": read2_dist,
        }
    )

    pd.to_csv("read_current_sweep_operating.csv")
