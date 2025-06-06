from typing import Literal, Tuple

import numpy as np

from nmem.analysis.utils import (
    filter_first,
)
from nmem.calculations.calculations import (
    calculate_heater_power,
    htron_critical_current,
)

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3


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


def get_fitting_points(
    x: np.ndarray, y: np.ndarray, ztotal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mid_idx = np.where(ztotal > np.nanmax(ztotal, axis=0) / 2)
    xfit, xfit_idx = np.unique(x[mid_idx[1]], return_index=True)
    yfit = y[mid_idx[0]][xfit_idx]
    return xfit, yfit


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
