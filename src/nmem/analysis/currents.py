from typing import Literal, Tuple

import numpy as np

from nmem.analysis.bit_error import (
    get_bit_error_rate,
    get_bit_error_rate_args,
)
from nmem.analysis.constants import (
    CRITICAL_TEMP,
    SUBSTRATE_TEMP,
)
from nmem.analysis.utils import filter_first, get_current_cell
from nmem.measurement.cells import CELLS


def calculate_channel_temperature(
    critical_temperature: float,
    substrate_temperature: float,
    ih: float,
    ih_max: float,
) -> float:
    N = 2.0
    beta = 1.25
    if ih_max == 0:
        raise ValueError("ih_max cannot be zero to avoid division by zero.")

    channel_temperature = (critical_temperature**4 - substrate_temperature**4) * (
        (ih / ih_max) ** N
    ) + substrate_temperature**4

    channel_temperature = np.maximum(channel_temperature, 0)

    temp_channel = np.power(channel_temperature, 0.25).astype(float)
    return temp_channel


def calculate_critical_current_zero(
    critical_temperature: float,
    substrate_temperature: float,
    critical_current_heater_off: float,
) -> np.ndarray:
    ic_zero = (
        critical_current_heater_off
        / (1 - (substrate_temperature / critical_temperature) ** 3) ** 2.1
    )
    return ic_zero


# def calculate_critical_current(T: np.ndarray, Tc: float, Ic0: float) -> np.ndarray:
#     return Ic0 * (1 - (T / Tc) ** (3 / 2))


def calculate_critical_current_temp(
    temp_array: np.ndarray, Tc: float, critical_current_zero: float
) -> np.ndarray:
    return critical_current_zero * (1 - (temp_array / Tc) ** (3)) ** (2.1)


def calculate_retrapping_current_temp(
    T: np.ndarray, Tc: float, critical_current_zero: float, retrap_ratio: float
) -> np.ndarray:
    Ir = retrap_ratio * critical_current_zero * (1 - (T / Tc)) ** (1 / 2)

    Ic = calculate_critical_current_temp(T, Tc, critical_current_zero)
    Ir = np.minimum(Ir, Ic)
    return Ir


def calculate_branch_currents(
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    critical_current_zero: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if any(T > Tc):
        raise ValueError("Temperature must be less than critical temperature.")
    ichr: np.ndarray = calculate_critical_current_temp(
        T, Tc, critical_current_zero * (1 - width_ratio)
    )
    ichl: np.ndarray = calculate_critical_current_temp(
        T, Tc, critical_current_zero * width_ratio
    )

    irhr: np.ndarray = calculate_retrapping_current_temp(
        T, Tc, critical_current_zero * (1 - width_ratio), retrap_ratio
    )
    irhl: np.ndarray = calculate_retrapping_current_temp(
        T, Tc, critical_current_zero * width_ratio, retrap_ratio
    )

    ichl = np.maximum(ichl, 0)
    irhl = np.maximum(irhl, 0)
    ichr = np.maximum(ichr, 0)
    irhr = np.maximum(irhr, 0)
    return ichl, irhl, ichr, irhr


def calculate_state_currents(
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    alpha: float,
    persistent_current: float,
    critical_current_zero: float,
) -> tuple:
    ichl, irhl, ichr, irhr = calculate_branch_currents(
        T, Tc, retrap_ratio, width_ratio, critical_current_zero
    )

    fa = ichr + irhl
    fb = ichl + irhr - persistent_current
    fc = (ichl - persistent_current) / alpha
    fd = fb - persistent_current

    fa = np.maximum(fa, 0)
    fb = np.maximum(fb, 0)
    fc = np.maximum(fc, 0)
    fd = np.maximum(fd, 0)
    return fa, fb, fc, fd


def extract_ic_vs_ih_data(data):
    """
    Extracts and returns heater currents, average current, std, and cell names from the data dict.
    """
    print(data.keys())
    ic_vs_ih = data["ic_vs_ih_data"]
    heater_currents = ic_vs_ih["heater_currents"][0, 0]
    avg_current = ic_vs_ih["avg_current"][0, 0]
    ystd = ic_vs_ih["ystd"][0, 0]
    cell_names = ic_vs_ih["cell_names"][0, 0]
    return heater_currents, avg_current, ystd, cell_names


def get_critical_current_heater_off(data_dict: dict) -> np.ndarray:
    cell = get_current_cell(data_dict)
    switching_current_heater_off = CELLS[cell]["max_critical_current"] * 1e6
    return switching_current_heater_off


def get_enable_read_current(data_dict: dict) -> float:
    return filter_first(data_dict.get("enable_read_current")) * 1e6


def get_enable_write_current(data_dict: dict) -> float:
    return filter_first(data_dict.get("enable_write_current")) * 1e6


def get_optimal_enable_read_current(current_cell: str) -> float:
    return CELLS[current_cell]["enable_read_current"] * 1e6


def get_optimal_enable_write_current(current_cell: str) -> float:
    return CELLS[current_cell]["enable_write_current"] * 1e6


def get_enable_current_sweep(data_dict: dict) -> np.ndarray:

    enable_current_array: np.ndarray = data_dict.get("x")[:, :, 0].flatten() * 1e6
    if len(enable_current_array) == 1:
        enable_current_array = data_dict.get("x")[:, 0].flatten() * 1e6

    if enable_current_array[0] == enable_current_array[1]:
        enable_current_array = data_dict.get("y")[:, :, 0].flatten() * 1e6

    return enable_current_array


def get_enable_currents_array(
    dict_list: list[dict], operation: Literal["read", "write"]
) -> np.ndarray:
    enable_currents = []
    for data_dict in dict_list:
        if operation == "read":
            enable_current = get_enable_read_current(data_dict)
        elif operation == "write":
            enable_current = get_enable_write_current(data_dict)
        enable_currents.append(enable_current)
    return np.array(enable_currents)


def get_write_currents(data_dict: dict) -> np.ndarray:
    write_currents = data_dict.get("write_current").flatten() * 1e6
    return write_currents


def get_read_currents(data_dict: dict) -> np.ndarray:
    read_currents = data_dict.get("y")[:, :, 0] * 1e6
    return read_currents.flatten()


def get_critical_currents_from_trace(dict_list: list) -> Tuple[np.ndarray, np.ndarray]:
    critical_currents = []
    critical_currents_std = []
    for data in dict_list:
        time = data.get("trace")[0, :]
        voltage = data.get("trace")[1, :]

        M = int(np.round(len(voltage), -2))
        if len(voltage) > M:
            voltage = voltage[:M]
            time = time[:M]
        else:
            voltage = np.concatenate([voltage, np.zeros(M - len(voltage))])
            time = np.concatenate([time, np.zeros(M - len(time))])

        current_time_trend = (
            data["vpp"]
            / 2
            / 10e3
            * (data["time_trend"][1, :])
            / (1 / (data["freq"] * 4))
            * 1e6
        )
        avg_critical_current = np.mean(current_time_trend)
        std_critical_current = np.std(current_time_trend)
        critical_currents.append(avg_critical_current)
        critical_currents_std.append(std_critical_current)
    critical_currents = np.array(critical_currents)
    critical_currents_std = np.array(critical_currents_std)
    return critical_currents, critical_currents_std


def get_max_enable_current(data_dict: dict) -> float:
    cell = get_current_cell(data_dict)
    return CELLS[cell]["x_intercept"]


def get_critical_current_intercept(data_dict: dict) -> float:
    cell = get_current_cell(data_dict)
    return CELLS[cell]["y_intercept"]


def get_channel_temperature(
    data_dict: dict, operation: Literal["read", "write"]
) -> float:
    if operation == "read":
        enable_current = get_enable_read_current(data_dict)
    elif operation == "write":
        enable_current = get_enable_write_current(data_dict)

    max_enable_current = get_max_enable_current(data_dict)

    channel_temp = calculate_channel_temperature(
        CRITICAL_TEMP, SUBSTRATE_TEMP, enable_current, max_enable_current
    )
    return channel_temp


def get_channel_temperature_sweep(data_dict: dict) -> np.ndarray:
    enable_currents = get_enable_current_sweep(data_dict)

    max_enable_current = get_max_enable_current(data_dict)
    channel_temps = calculate_channel_temperature(
        CRITICAL_TEMP, SUBSTRATE_TEMP, enable_currents, max_enable_current
    )
    return channel_temps


def get_write_current(data_dict: dict) -> float:
    if data_dict.get("write_current").shape[1] == 1:
        return filter_first(data_dict.get("write_current")) * 1e6
    if data_dict.get("write_current").shape[1] > 1:
        return data_dict.get("write_current")[0, 0] * 1e6


def get_read_current(data_dict: dict) -> float:
    if data_dict.get("read_current").shape[1] == 1:
        return filter_first(data_dict.get("read_current")) * 1e6


def get_state_current_markers_list(
    dict_list: list[dict],
    current_sweep: Literal["read_current", "enable_write_current"],
) -> list[np.ndarray]:
    state_current_markers_list = []
    for data_dict in dict_list:
        state_current_markers = get_state_current_markers(data_dict, current_sweep)
        state_current_markers_list.append(state_current_markers)
    return state_current_markers_list


def get_state_current_markers(
    data_dict: dict, current_sweep: Literal["read_current", "enable_write_current"]
) -> np.ndarray:
    if current_sweep == "read_current":
        currents = get_read_currents(data_dict)
    if current_sweep == "enable_write_current":
        currents = get_enable_current_sweep(data_dict)
    bit_error_rate = get_bit_error_rate(data_dict)
    berargs = get_bit_error_rate_args(bit_error_rate)
    state_current_markers = np.zeros((2, 4))
    for arg in berargs:
        if arg is not np.nan:
            state_current_markers[0, berargs.index(arg)] = currents[arg]
            state_current_markers[1, berargs.index(arg)] = bit_error_rate[arg]
        else:
            state_current_markers[0, berargs.index(arg)] = np.nan
            state_current_markers[1, berargs.index(arg)] = np.nan

    return state_current_markers


def get_state_currents_measured_array(
    dict_list: list[dict], current_sweep: str
) -> np.ndarray:
    temps = []
    state_currents = []
    for data_dict in dict_list:
        temp, state_current = get_state_currents_measured(data_dict, current_sweep)
        temps.append(temp)
        state_currents.append(state_current)
    return np.array(temps), np.array(state_currents)


def get_state_currents_array(dict_list: list[dict]) -> np.ndarray:
    state_currents = []
    for data_dict in dict_list:
        get_state_currents_measured(data_dict)
        state_currents.append(get_state_currents_measured(data_dict))
    return np.array(state_currents)


def get_state_currents_measured(
    data_dict: dict,
    current_sweep: Literal["enable_write_current", "enable_read_current"],
) -> Tuple[np.ndarray, np.ndarray]:
    bit_error_rate = get_bit_error_rate(data_dict)
    nominal_edge1, nominal_edge2, inverting_edge1, inverting_edge2 = (
        get_bit_error_rate_args(bit_error_rate)
    )
    if current_sweep == "enable_write_current":
        temperature = get_channel_temperature(data_dict, "write")
    else:
        temperature = get_channel_temperature(data_dict, "read")

    read_currents = get_read_currents(data_dict)
    if nominal_edge1 is not np.nan:
        nominal_state0_current = read_currents[nominal_edge2]
        nominal_state1_current = read_currents[nominal_edge1]
    else:
        nominal_state0_current = np.nan
        nominal_state1_current = np.nan
    if inverting_edge1 is not np.nan:
        inverting_state0_current = read_currents[inverting_edge2]
        inverting_state1_current = read_currents[inverting_edge1]
    else:
        inverting_state0_current = np.nan
        inverting_state1_current = np.nan
    temp = np.array(temperature)
    state_currents = np.array(
        [
            nominal_state0_current,
            nominal_state1_current,
            inverting_state0_current,
            inverting_state1_current,
        ]
    )
    return temp, state_currents
