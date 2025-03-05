import numpy as np
import pandas as pd

from nmem.analysis.analysis import (
    CELLS,
    CRITICAL_TEMP,
    SUBSTRATE_TEMP,
    WIDTH,
    calculate_channel_temperature,
    calculate_critical_current_temp,
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_read_currents,
    get_write_current,
    import_directory,
)
from nmem.analysis.write_current_sweep_sub import calculate_persistent_currents

IRM = 727
IRHL_TR = 110
CRITICAL_CURRENT_ZERO = 1250


def calculate_inductance_ratio(state0, state1, ic0):
    alpha = (ic0 - state1) / (state0 - state1)
    # alpha_test = 1 - ((critical_current_right - persistent_current_est) / ic)
    # alpha_test2 = (critical_current_left - persistent_current_est) / ic2

    return alpha


def get_state_currents(data_dict: list[dict], left_retrapping_current: float = 100):
    write_current = get_write_current(data_dict)
    bit_error_rate = get_bit_error_rate(data_dict)
    berargs = get_bit_error_rate_args(bit_error_rate)
    read_currents = get_read_currents(data_dict)

    ic = np.nan
    ic2 = np.nan

    if not np.isnan(berargs[0]) and write_current < left_retrapping_current:
        ic = read_currents[berargs[0]]
    if not np.isnan(berargs[2]) and write_current > left_retrapping_current:
        ic = read_currents[berargs[3]]
    if not np.isnan(berargs[1]):
        ic2 = read_currents[berargs[1]]
    if not np.isnan(berargs[3]):
        ic2 = read_currents[berargs[2]]

    return ic, ic2, write_current


def process_dict(data_dict):
    ic, ic2, write_current_list = get_state_currents(data_dict)
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
    )[0]
    return (
        ic,
        ic2,
        write_current_list,
        read_temperature,
        write_temperature,
        critical_current_channel,
    )


if __name__ == "__main__":
    # Import
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\read_current_sweep_write_current2\write_current_sweep_C3"
    )

    # Preprocess
    results = []
    for data_dict in dict_list:
        (
            ic,
            ic2,
            write_current,
            read_temperature,
            write_temperature,
            critical_current_channel,
        ) = process_dict(data_dict)

        critical_current_left = critical_current_channel * WIDTH
        critical_current_right = critical_current_channel * (1 - WIDTH)
        read_current_difference = ic2 - ic
        read1_dist = np.abs(ic - IRM)
        read2_dist = np.abs(ic2 - IRM)
        expected_persistent_current = calculate_persistent_currents(
            write_current, IRHL_TR
        )
        results.append(
            {
                "Write Current": write_current,
                "Read Current": ic,
                "Read Current 2": ic2,
                "Channel Critical Current": critical_current_channel,
                "Channel Critical Current Left": critical_current_left,
                "Channel Critical Current Right": critical_current_right,
                "Persistent Current": expected_persistent_current,
                "Read Current Difference": read_current_difference,
                "Read Current 1 Distance": read1_dist,
                "Read Current 2 Distance": read2_dist,
            }
        )

    # Print table
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("read_current_sweep_operating_calculation.csv")
