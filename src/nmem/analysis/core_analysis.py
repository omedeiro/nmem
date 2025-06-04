import collections
import collections.abc
import os
from typing import Any, Literal, Tuple

import numpy as np
import scipy.io as sio

from nmem.calculations.calculations import (
    calculate_heater_power,
    htron_critical_current,
)
from nmem.measurement.cells import CELLS

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


def build_array(
    data_dict: dict, parameter_z: Literal["total_switches_norm"]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data_dict.get("total_switches_norm") is None:
        data_dict["total_switches_norm"] = get_total_switches_norm(data_dict)
    x: np.ndarray = data_dict.get("x")[0][:, 0] * 1e6
    y: np.ndarray = data_dict.get("y")[0][:, 0] * 1e6
    z: np.ndarray = data_dict.get(parameter_z)

    xlength: int = filter_first(data_dict.get("sweep_x_len", len(x)))
    ylength: int = filter_first(data_dict.get("sweep_y_len", len(y)))

    # X, Y reversed in reshape
    zarray = z.reshape((ylength, xlength), order="F")
    return x, y, zarray


def filter_plateau(
    xfit: np.ndarray, yfit: np.ndarray, plateau_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    xfit = np.where(yfit < plateau_height, xfit, np.nan)
    yfit = np.where(yfit < plateau_height, yfit, np.nan)

    # Remove nans
    xfit = xfit[~np.isnan(xfit)]
    yfit = yfit[~np.isnan(yfit)]

    return xfit, yfit


def filter_first(value) -> Any:
    if isinstance(value, collections.abc.Iterable) and not isinstance(
        value, (str, bytes)
    ):
        return np.asarray(value).flatten()[0]
    return value


def convert_cell_to_coordinates(cell: str) -> tuple:
    """Converts a cell name like 'A1' to coordinates (x, y)."""
    column_letter = cell[0]
    row_number = int(cell[1:]) - 1
    column_number = ord(column_letter) - ord("A")
    return column_number, row_number


def calculate_bit_error_rate(data_dict: dict) -> np.ndarray:
    num_meas = data_dict.get("num_meas")[0][0]
    w1r0 = data_dict.get("write_1_read_0")[0].flatten() / num_meas
    w0r1 = data_dict.get("write_0_read_1")[0].flatten() / num_meas
    ber = (w1r0 + w0r1) / 2
    return ber


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


def get_current_cell(data_dict: dict) -> str:
    cell = filter_first(data_dict.get("cell"))
    if cell is None:
        cell = filter_first(data_dict.get("sample_name"))[-2:]
    return cell


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


def get_enable_write_width(data_dict: dict) -> float:
    return filter_first(data_dict.get("enable_write_width"))


def get_read_width(data_dict: dict) -> float:
    return filter_first(data_dict.get("read_width"))


def get_write_width(data_dict: dict) -> float:
    return filter_first(data_dict.get("write_width"))


def get_average_response(cell_dict):
    slope_list = []
    intercept_list = []
    for value in cell_dict.values():
        slope_list.append(value.get("slope"))
        intercept_list.append(value.get("y_intercept"))

    slope = np.mean(slope_list)
    intercept = np.mean(intercept_list)
    return slope, intercept



def get_text_from_bit(bit: str) -> str:
    if bit == "0":
        return "WR0"
    elif bit == "1":
        return "WR1"
    elif bit == "N":
        return ""
    elif bit == "R":
        return "RD"
    elif bit == "E":
        return "Read \nEnable"
    elif bit == "W":
        return "Write \nEnable"
    else:
        return None


def get_text_from_bit_v2(bit: str) -> str:
    if bit == "0":
        return "WR0"
    elif bit == "1":
        return "WR1"
    elif bit == "N":
        return ""
    elif bit == "R":
        return "RD"
    elif bit == "E":
        return "ER"
    elif bit == "W":
        return "EW"
    elif bit == "z":
        return "RD0"
    elif bit == "Z":
        return "W0R1"
    elif bit == "o":
        return "RD1"
    elif bit == "O":
        return "W1R0"
    else:
        return None


def get_text_from_bit_v3(bit: str) -> str:
    if bit == "0":
        return "0"
    elif bit == "1":
        return "1"
    elif bit == "N":
        return ""
    elif bit == "R":
        return ""
    elif bit == "E":
        return ""
    elif bit == "W":
        return ""
    else:
        return None


def get_total_switches_norm(data_dict: dict) -> np.ndarray:
    num_meas = data_dict.get("num_meas")[0][0]
    w0r1 = data_dict.get("write_0_read_1").flatten()
    w1r0 = num_meas - data_dict.get("write_1_read_0").flatten()
    total_switches_norm = (w0r1 + w1r0) / (num_meas * 2)
    return total_switches_norm


def get_write_currents(data_dict: dict) -> np.ndarray:
    write_currents = data_dict.get("write_current").flatten() * 1e6
    return write_currents


def get_read_currents(data_dict: dict) -> np.ndarray:
    read_currents = data_dict.get("y")[:, :, 0] * 1e6
    return read_currents.flatten()


def get_bit_error_rate(data_dict: dict) -> np.ndarray:
    return data_dict.get("bit_error_rate").flatten()


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
        print(f"length of current_time_trend: {len(current_time_trend[0])}")
        avg_critical_current = np.mean(current_time_trend)
        std_critical_current = np.std(current_time_trend)
        critical_currents.append(avg_critical_current)
        critical_currents_std.append(std_critical_current)
    critical_currents = np.array(critical_currents)
    critical_currents_std = np.array(critical_currents_std)
    return critical_currents, critical_currents_std


def get_fitting_points(
    x: np.ndarray, y: np.ndarray, ztotal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mid_idx = np.where(ztotal > np.nanmax(ztotal, axis=0) / 2)
    xfit, xfit_idx = np.unique(x[mid_idx[1]], return_index=True)
    yfit = y[mid_idx[0]][xfit_idx]
    return xfit, yfit


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


def get_operating_points(data_dict: dict) -> np.ndarray:
    berargs = get_bit_error_rate_args(get_bit_error_rate(data_dict))
    nominal_operating_point = np.mean([berargs[0], berargs[1]])
    inverting_operating_point = np.mean([berargs[2], berargs[3]])
    return nominal_operating_point, inverting_operating_point


def get_bit_error_rate_args(bit_error_rate: np.ndarray) -> list:
    nominal_args = np.argwhere(bit_error_rate < 0.45)
    inverting_args = np.argwhere(bit_error_rate > 0.55)

    if len(inverting_args) > 0:
        inverting_arg1 = inverting_args[0][0]
        inverting_arg2 = inverting_args[-1][0]
    else:
        inverting_arg1 = np.nan
        inverting_arg2 = np.nan

    if len(nominal_args) > 0:
        nominal_arg1 = nominal_args[0][0]
        nominal_arg2 = nominal_args[-1][0]
    else:
        nominal_arg1 = np.nan
        nominal_arg2 = np.nan

    return nominal_arg1, nominal_arg2, inverting_arg1, inverting_arg2


def get_voltage_trace_data(
    data_dict: dict,
    trace_name: Literal["trace_chan_in", "trace_chan_out", "trace_enab"],
    trace_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if data_dict.get(trace_name).ndim == 2:
        x = data_dict[trace_name][0] * 1e6
        y = data_dict[trace_name][1] * 1e3
    else:
        x = data_dict[trace_name][0][:, trace_index] * 1e6
        y = data_dict[trace_name][1][:, trace_index] * 1e3
    return x, y





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




def initialize_dict(array_size: tuple) -> dict:
    return {
        "write_current": np.zeros(array_size),
        "write_current_norm": np.zeros(array_size),
        "read_current": np.zeros(array_size),
        "read_current_norm": np.zeros(array_size),
        "slope": np.zeros(array_size),
        "y_intercept": np.zeros(array_size),
        "x_intercept": np.zeros(array_size),
        "resistance": np.zeros(array_size),
        "bit_error_rate": np.zeros(array_size),
        "max_critical_current": np.zeros(array_size),
        "enable_write_current": np.zeros(array_size),
        "enable_write_current_norm": np.zeros(array_size),
        "enable_read_current": np.zeros(array_size),
        "enable_read_current_norm": np.zeros(array_size),
        "enable_write_power": np.zeros(array_size),
        "enable_read_power": np.zeros(array_size),
    }


def polygon_under_graph(x, y, y2=0.0):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], y2), *zip(x, y), (x[-1], y2)]


def polygon_nominal(x: np.ndarray, y: np.ndarray) -> list:
    y = np.copy(y)
    y[y > 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def polygon_inverting(x: np.ndarray, y: np.ndarray) -> list:
    y = np.copy(y)
    y[y < 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def process_cell(cell: dict, param_dict: dict, x: int, y: int) -> dict:
    param_dict["write_current"][y, x] = cell["write_current"] * 1e6
    param_dict["read_current"][y, x] = cell["read_current"] * 1e6
    param_dict["enable_write_current"][y, x] = cell["enable_write_current"] * 1e6
    param_dict["enable_read_current"][y, x] = cell["enable_read_current"] * 1e6
    param_dict["slope"][y, x] = cell["slope"]
    param_dict["y_intercept"][y, x] = cell["y_intercept"]
    param_dict["resistance"][y, x] = cell["resistance_cryo"]
    param_dict["bit_error_rate"][y, x] = cell.get("min_bit_error_rate", np.nan)
    param_dict["max_critical_current"][y, x] = (
        cell.get("max_critical_current", np.nan) * 1e6
    )
    if cell["y_intercept"] != 0:
        write_critical_current = htron_critical_current(
            cell["enable_write_current"] * 1e6, cell["slope"], cell["y_intercept"]
        )
        read_critical_current = htron_critical_current(
            cell["enable_read_current"] * 1e6, cell["slope"], cell["y_intercept"]
        )
        param_dict["x_intercept"][y, x] = -cell["y_intercept"] / cell["slope"]
        param_dict["write_current_norm"][y, x] = (
            cell["write_current"] * 1e6 / write_critical_current
        )
        param_dict["read_current_norm"][y, x] = (
            cell["read_current"] * 1e6 / read_critical_current
        )
        param_dict["enable_write_power"][y, x] = calculate_heater_power(
            cell["enable_write_current"], cell["resistance_cryo"]
        )
        param_dict["enable_write_current_norm"][y, x] = (
            cell["enable_write_current"] * 1e6 / param_dict["x_intercept"][y, x]
        )
        param_dict["enable_read_power"][y, x] = calculate_heater_power(
            cell["enable_read_current"], cell["resistance_cryo"]
        )
        param_dict["enable_read_current_norm"][y, x] = (
            cell["enable_read_current"] * 1e6 / param_dict["x_intercept"][y, x]
        )
    return param_dict


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

def filter_nan(x, y):
    mask = np.isnan(y)
    x = x[~mask]
    y = y[~mask]
    return x, y



