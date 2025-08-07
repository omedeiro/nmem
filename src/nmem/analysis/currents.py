from typing import Literal, Tuple, Union
from functools import lru_cache

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

# Constants
MICROAMP_CONVERSION = 1e6
DEFAULT_N = 2.0
DEFAULT_BETA = 1.25


# Core calculation functions
def calculate_channel_temperature(
    critical_temperature: float,
    substrate_temperature: float,
    ih: Union[float, np.ndarray],
    ih_max: float,
    N: float = DEFAULT_N,
) -> Union[float, np.ndarray]:
    """Calculate channel temperature with improved efficiency."""
    if ih_max == 0:
        raise ValueError("ih_max cannot be zero to avoid division by zero.")

    # Vectorized calculation
    ratio = (ih / ih_max) ** N
    temp_diff = critical_temperature**4 - substrate_temperature**4
    channel_temp = temp_diff * ratio + substrate_temperature**4
    channel_temp = np.maximum(channel_temp, 0)

    return np.power(channel_temp, 0.25).astype(float)


def calculate_critical_current_zero(
    critical_temperature: float,
    substrate_temperature: float,
    critical_current_heater_off: float,
) -> float:
    """Calculate critical current at zero temperature."""
    return (
        critical_current_heater_off
        / (1 - (substrate_temperature / critical_temperature) ** 3) ** 2.1
    )


def calculate_critical_current_temp(
    temp_array: np.ndarray, Tc: float, critical_current_zero: float
) -> np.ndarray:
    """Calculate critical current as function of temperature."""
    return critical_current_zero * (1 - (temp_array / Tc) ** 3) ** 2.1


def calculate_retrapping_current_temp(
    T: np.ndarray, Tc: float, critical_current_zero: float, retrap_ratio: float
) -> np.ndarray:
    """Calculate retrapping current with temperature dependence."""
    Ir = retrap_ratio * critical_current_zero * (1 - (T / Tc)) ** 0.5
    Ic = calculate_critical_current_temp(T, Tc, critical_current_zero)
    return np.minimum(Ir, Ic)


def calculate_branch_currents(
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    critical_current_zero: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate all branch currents efficiently."""
    if np.any(T > Tc):
        raise ValueError("Temperature must be less than critical temperature.")

    # Pre-calculate ratios
    ic_zero_left = critical_current_zero * width_ratio
    ic_zero_right = critical_current_zero * (1 - width_ratio)

    # Calculate all currents at once
    ichl = calculate_critical_current_temp(T, Tc, ic_zero_left)
    ichr = calculate_critical_current_temp(T, Tc, ic_zero_right)
    irhl = calculate_retrapping_current_temp(T, Tc, ic_zero_left, retrap_ratio)
    irhr = calculate_retrapping_current_temp(T, Tc, ic_zero_right, retrap_ratio)

    # Apply lower bound to all at once
    return tuple(np.maximum(curr, 0) for curr in (ichl, irhl, ichr, irhr))


def calculate_state_currents(
    T: np.ndarray,
    Tc: float,
    retrap_ratio: float,
    width_ratio: float,
    alpha: float,
    persistent_current: float,
    critical_current_zero: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate state currents with improved efficiency."""
    ichl, irhl, ichr, irhr = calculate_branch_currents(
        T, Tc, retrap_ratio, width_ratio, critical_current_zero
    )

    fa = ichr + irhl
    fb = ichl + irhr - persistent_current
    fc = (ichl - persistent_current) / alpha
    fd = fb - persistent_current

    # Apply lower bound to all at once
    return tuple(np.maximum(curr, 0) for curr in (fa, fb, fc, fd))


# Utility functions for data extraction
def extract_ic_vs_ih_data(
    data: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract IC vs IH data from data dictionary."""
    ic_vs_ih = data["ic_vs_ih_data"]
    return (
        ic_vs_ih["heater_currents"][0, 0],
        ic_vs_ih["avg_current"][0, 0],
        ic_vs_ih["ystd"][0, 0],
        ic_vs_ih["cell_names"][0, 0],
    )


@lru_cache(maxsize=128)
def _get_cell_property(cell: str, property_name: str) -> float:
    """Cached cell property getter to avoid repeated lookups."""
    return CELLS[cell][property_name]


def _get_current_with_conversion(
    data_dict: dict, key: str, conversion_factor: float = MICROAMP_CONVERSION
) -> float:
    """Generic function to get current values with unit conversion."""
    value = filter_first(data_dict.get(key))
    return value * conversion_factor if value is not None else 0.0


# Simplified current getter functions
def get_critical_current_heater_off(data_dict: dict) -> float:
    """Get critical current when heater is off."""
    cell = get_current_cell(data_dict)
    return _get_cell_property(cell, "max_critical_current") * MICROAMP_CONVERSION


def get_enable_read_current(data_dict: dict) -> float:
    """Get enable read current."""
    return _get_current_with_conversion(data_dict, "enable_read_current")


def get_enable_write_current(data_dict: dict) -> float:
    """Get enable write current."""
    return _get_current_with_conversion(data_dict, "enable_write_current")


def get_optimal_enable_current(
    current_cell: str, operation: Literal["read", "write"]
) -> float:
    """Get optimal enable current for read or write operations."""
    key = f"enable_{operation}_current"
    return _get_cell_property(current_cell, key) * MICROAMP_CONVERSION


# Legacy wrapper functions for backward compatibility
def get_optimal_enable_read_current(current_cell: str) -> float:
    return get_optimal_enable_current(current_cell, "read")


def get_optimal_enable_write_current(current_cell: str) -> float:
    return get_optimal_enable_current(current_cell, "write")


def get_enable_current_sweep(data_dict: dict) -> np.ndarray:
    """Extract enable current sweep with improved logic."""
    # Try different data locations
    for key_path in [
        ("x", (slice(None), slice(None), 0)),
        ("x", (slice(None), 0)),
        ("y", (slice(None), slice(None), 0)),
    ]:
        try:
            key, indices = key_path
            enable_current_array = (
                data_dict.get(key)[indices].flatten() * MICROAMP_CONVERSION
            )

            # Check if we have valid data (not all the same value)
            if (
                len(enable_current_array) > 1
                and enable_current_array[0] != enable_current_array[1]
            ):
                return enable_current_array
        except (IndexError, TypeError):
            continue

    # Fallback
    return np.array([])


def get_enable_currents_array(
    dict_list: list[dict], operation: Literal["read", "write"]
) -> np.ndarray:
    """Get enable currents array with vectorized approach."""
    getter_func = (
        get_enable_read_current if operation == "read" else get_enable_write_current
    )
    return np.array([getter_func(data_dict) for data_dict in dict_list])


def get_currents_from_data(
    data_dict: dict, current_type: Literal["write", "read"]
) -> np.ndarray:
    """Unified function to get currents from data dictionary."""
    if current_type == "write":
        return (
            data_dict.get("write_current", np.array([])).flatten() * MICROAMP_CONVERSION
        )
    else:  # read
        return (
            data_dict.get("y", np.array([[[]]]))[:, :, 0] * MICROAMP_CONVERSION
        ).flatten()


# Simplified current getter functions
def get_write_currents(data_dict: dict) -> np.ndarray:
    return get_currents_from_data(data_dict, "write")


def get_read_currents(data_dict: dict) -> np.ndarray:
    return get_currents_from_data(data_dict, "read")


def get_critical_currents_from_trace(
    dict_list: list[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract critical currents from trace data with improved efficiency."""
    critical_currents = []
    critical_currents_std = []

    for data in dict_list:
        trace = data.get("trace", np.array([[], []]))
        if trace.size == 0:
            critical_currents.extend([np.nan])
            critical_currents_std.extend([np.nan])
            continue

        time, voltage = trace[0, :], trace[1, :]

        # Efficient array resizing
        M = int(np.round(len(voltage), -2))
        if len(voltage) != M:
            if len(voltage) > M:
                voltage, time = voltage[:M], time[:M]
            else:
                padding = M - len(voltage)
                voltage = np.pad(voltage, (0, padding), mode="constant")
                time = np.pad(time, (0, padding), mode="constant")

        # Vectorized current calculation
        current_time_trend = (
            data["vpp"]
            / 2
            / 10e3
            * data["time_trend"][1, :]
            / (1 / (data["freq"] * 4))
            * MICROAMP_CONVERSION
        )

        critical_currents.append(np.mean(current_time_trend))
        critical_currents_std.append(np.std(current_time_trend))

    return np.array(critical_currents), np.array(critical_currents_std)


def get_cell_properties(data_dict: dict) -> Tuple[float, float]:
    """Get max enable current and critical current intercept for a cell."""
    cell = get_current_cell(data_dict)
    return (
        _get_cell_property(cell, "x_intercept"),
        _get_cell_property(cell, "y_intercept"),
    )


def get_max_enable_current(data_dict: dict) -> float:
    """Get maximum enable current for a cell."""
    return get_cell_properties(data_dict)[0]


def get_critical_current_intercept(data_dict: dict) -> float:
    """Get critical current intercept for a cell."""
    return get_cell_properties(data_dict)[1]


def get_channel_temperature(
    data_dict: dict, operation: Literal["read", "write"]
) -> float:
    """Calculate channel temperature for given operation."""
    enable_current = (
        get_enable_read_current if operation == "read" else get_enable_write_current
    )(data_dict)
    max_enable_current = get_max_enable_current(data_dict)

    return calculate_channel_temperature(
        CRITICAL_TEMP, SUBSTRATE_TEMP, enable_current, max_enable_current
    )


def get_channel_temperature_sweep(data_dict: dict) -> np.ndarray:
    """Calculate channel temperatures for current sweep."""
    enable_currents = get_enable_current_sweep(data_dict)
    max_enable_current = get_max_enable_current(data_dict)

    return calculate_channel_temperature(
        CRITICAL_TEMP, SUBSTRATE_TEMP, enable_currents, max_enable_current
    )


def get_current_from_data_shape(data_dict: dict, key: str) -> float:
    """Extract current based on data shape with improved logic."""
    current_data = data_dict.get(key, np.array([[]]))

    if current_data.size == 0:
        return 0.0

    if current_data.shape[1] == 1:
        return filter_first(current_data) * MICROAMP_CONVERSION
    elif current_data.shape[1] > 1:
        return current_data[0, 0] * MICROAMP_CONVERSION

    return 0.0


def get_write_current(data_dict: dict) -> float:
    """Get single write current value."""
    return get_current_from_data_shape(data_dict, "write_current")


def get_read_current(data_dict: dict) -> float:
    """Get single read current value."""
    return get_current_from_data_shape(data_dict, "read_current")


def get_state_current_markers_list(
    dict_list: list[dict],
    current_sweep: Literal["read_current", "enable_write_current"],
) -> list[np.ndarray]:
    """Get state current markers for multiple data dictionaries."""
    return [
        get_state_current_markers(data_dict, current_sweep) for data_dict in dict_list
    ]


def get_state_current_markers(
    data_dict: dict, current_sweep: Literal["read_current", "enable_write_current"]
) -> np.ndarray:
    """Get state current markers with improved efficiency."""
    # Get currents based on sweep type
    currents = (
        get_read_currents(data_dict)
        if current_sweep == "read_current"
        else get_enable_current_sweep(data_dict)
    )

    bit_error_rate = get_bit_error_rate(data_dict)
    berargs = get_bit_error_rate_args(bit_error_rate)

    # Vectorized marker calculation
    state_current_markers = np.full((2, 4), np.nan)
    valid_indices = [i for i, arg in enumerate(berargs) if arg is not np.nan]

    for i in valid_indices:
        arg = berargs[i]
        state_current_markers[0, i] = currents[arg]
        state_current_markers[1, i] = bit_error_rate[arg]

    return state_current_markers


def get_state_currents_measured_array(
    dict_list: list[dict], current_sweep: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Get measured state currents for multiple data dictionaries."""
    results = [
        get_state_currents_measured(data_dict, current_sweep) for data_dict in dict_list
    ]
    temps, state_currents = zip(*results)
    return np.array(temps), np.array(state_currents)


def get_state_currents_array(dict_list: list[dict]) -> np.ndarray:
    """Get state currents array - fixed to avoid duplicate calls."""
    return np.array(
        [
            get_state_currents_measured(data_dict, "enable_write_current")[1]
            for data_dict in dict_list
        ]
    )


def _extract_state_currents_from_edges(
    read_currents: np.ndarray, edge1: Union[int, float], edge2: Union[int, float]
) -> Tuple[float, float]:
    """Helper function to extract state currents from edge indices."""
    if edge1 is not np.nan and edge2 is not np.nan:
        return read_currents[edge2], read_currents[edge1]
    return np.nan, np.nan


def get_state_currents_measured(
    data_dict: dict,
    current_sweep: Literal["enable_write_current", "enable_read_current"],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract measured state currents with improved efficiency."""
    bit_error_rate = get_bit_error_rate(data_dict)
    nominal_edge1, nominal_edge2, inverting_edge1, inverting_edge2 = (
        get_bit_error_rate_args(bit_error_rate)
    )

    # Get temperature based on current sweep type
    operation = "write" if current_sweep == "enable_write_current" else "read"
    temperature = get_channel_temperature(data_dict, operation)

    read_currents = get_read_currents(data_dict)

    # Extract state currents using helper function
    nominal_state0_current, nominal_state1_current = _extract_state_currents_from_edges(
        read_currents, nominal_edge1, nominal_edge2
    )
    inverting_state0_current, inverting_state1_current = (
        _extract_state_currents_from_edges(
            read_currents, inverting_edge1, inverting_edge2
        )
    )

    return (
        np.array(temperature),
        np.array(
            [
                nominal_state0_current,
                nominal_state1_current,
                inverting_state0_current,
                inverting_state1_current,
            ]
        ),
    )
