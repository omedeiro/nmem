"""Data processing utilities for SPICE simulation results."""

import os
from typing import Tuple, Dict, Any, List
import numpy as np
import scipy.io as sio
import ltspice
from ..utils.constants import VOLTAGE_THRESHOLD


def get_persistent_current(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    """Extract persistent current from LTspice simulation data.

    Args:
        ltspice_data: Parsed LTspice data object
        case: Case number for parametric sweeps

    Returns:
        Array containing persistent current value in µA
    """
    signal_l = ltspice_data.get_data("Ix(HL:drain)", case=case)
    signal_r = ltspice_data.get_data("Ix(HR:drain)", case=case)
    return np.array([np.abs(signal_r[-1] - signal_l[-1]) / 2 * 1e6])


def get_write_current(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    """Extract write current from LTspice simulation data.

    Args:
        ltspice_data: Parsed LTspice data object
        case: Case number for parametric sweeps

    Returns:
        Maximum write current in µA
    """
    signal = ltspice_data.get_data("I(R2)", case=case)
    return np.max(signal) * 1e6


def get_max_output(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    """Extract maximum output voltage from LTspice simulation data.

    Args:
        ltspice_data: Parsed LTspice data object
        case: Case number for parametric sweeps

    Returns:
        Maximum output voltage
    """
    signal = ltspice_data.get_data("V(out)", case=case)
    return np.max(signal)


def process_read_data(ltspice_data: ltspice.Ltspice) -> List[Dict[str, Any]]:
    """Process LTspice simulation data and extract key metrics.

    This function is used by the analysis modules to extract standardized
    data from SPICE simulation results.

    Args:
        ltspice_data: Parsed LTspice data object

    Returns:
        List of dictionaries containing processed data for each case
    """
    processed_data = []

    # Handle single case or multiple cases
    num_cases = getattr(ltspice_data, "case_count", 1)

    for case in range(num_cases):
        case_data = {}

        # Extract time data
        time = ltspice_data.get_time(case=case)
        case_data["time"] = time

        # Extract current signals if available
        try:
            case_data["left_current"] = ltspice_data.get_data("Ix(HL:drain)", case=case)
        except:
            try:
                case_data["left_current"] = ltspice_data.get_data(
                    "Ix(hl:drain)", case=case
                )
            except:
                case_data["left_current"] = np.zeros_like(time)

        try:
            case_data["right_current"] = ltspice_data.get_data(
                "Ix(HR:drain)", case=case
            )
        except:
            try:
                case_data["right_current"] = ltspice_data.get_data(
                    "Ix(hr:drain)", case=case
                )
            except:
                case_data["right_current"] = np.zeros_like(time)

        # Extract write current
        try:
            write_current = ltspice_data.get_data("I(R2)", case=case)
            case_data["write_current"] = [np.max(write_current)]
        except:
            case_data["write_current"] = [0.0]

        # Extract read current
        try:
            read_current = ltspice_data.get_data("I(R2)", case=case)
            case_data["read_current"] = [np.max(read_current)]
        except:
            case_data["read_current"] = np.zeros_like(time)

        # Extract output voltage
        try:
            case_data["output_voltage"] = ltspice_data.get_data("V(out)", case=case)
        except:
            case_data["output_voltage"] = np.zeros_like(time)

        # Calculate persistent current
        try:
            persistent = (
                np.abs(case_data["right_current"][-1] - case_data["left_current"][-1])
                / 2
            )
            case_data["persistent_current"] = [persistent]
        except:
            case_data["persistent_current"] = [0.0]

        processed_data.append(case_data)

    return processed_data


def get_processed_state_currents(
    data_dict: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Process read state currents from simulation data.

    Args:
        data_dict: Dictionary containing read voltage and current data

    Returns:
        Tuple of (zero_current, one_current)
    """
    read_zero_voltage = data_dict.get("read_zero_voltage")
    read_one_voltage = data_dict.get("read_one_voltage")

    if any(read_zero_voltage > VOLTAGE_THRESHOLD):
        zero_switch = np.argwhere(read_zero_voltage > VOLTAGE_THRESHOLD).flat[0]
    else:
        zero_switch = 0

    if any(read_one_voltage > VOLTAGE_THRESHOLD):
        one_switch = np.argwhere(read_one_voltage > VOLTAGE_THRESHOLD).flat[0]
    else:
        one_switch = 0

    zero_current = data_dict.get("read_current")[zero_switch]
    one_current = data_dict.get("read_current")[one_switch]
    return zero_current, one_current


def process_data_dict_sweep(
    data_dict: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process sweep data dictionary.

    Args:
        data_dict: Dictionary containing sweep simulation data

    Returns:
        Tuple of (zero_currents, one_currents, read_currents)
    """
    zero_currents = []
    one_currents = []
    read_currents = []

    for key in data_dict.keys():
        zero_current, one_current = get_processed_state_currents(data_dict[key])
        zero_currents.append(zero_current)
        one_currents.append(one_current)
        read_currents.append(data_dict[key].get("read_current"))

    return np.array(zero_currents), np.array(one_currents), np.array(read_currents)


def process_data_dict_write_sweep(
    data_dict: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Process write sweep data dictionary.

    Args:
        data_dict: Dictionary containing write sweep simulation data

    Returns:
        Tuple of (write_currents, persistent_currents)
    """
    write_currents = []
    persistent_currents = []

    for key in data_dict.keys():
        write_current = data_dict[key].get("write_current")
        persistent_current = data_dict[key].get("persistent_current")
        write_currents.append(write_current)
        persistent_currents.append(persistent_current)

    return np.array(write_currents), np.array(persistent_currents)


def get_write_sweep_data(file_path: str) -> Dict[str, Any]:
    """Load write sweep data from file.

    Args:
        file_path: Path to the data file

    Returns:
        Dictionary containing loaded data
    """
    data_dict = {}

    try:
        # Try loading as .mat file first
        mat_data = sio.loadmat(file_path)
        data_dict = {
            key: mat_data[key] for key in mat_data.keys() if not key.startswith("__")
        }
    except:
        # If that fails, try other formats or implement custom loading
        pass

    return data_dict


def get_bit_error_rate(
    zero_currents: np.ndarray, one_currents: np.ndarray, read_currents: np.ndarray
) -> np.ndarray:
    """Calculate bit error rate from current measurements.

    Args:
        zero_currents: Array of zero state currents
        one_currents: Array of one state currents
        read_currents: Array of read currents

    Returns:
        Array of bit error rates
    """
    ber = []
    for i, read_current in enumerate(read_currents):
        # Simple threshold-based BER calculation
        threshold = (zero_currents[i] + one_currents[i]) / 2
        errors = np.sum(
            np.abs(read_current - threshold)
            < np.abs(zero_currents[i] - one_currents[i]) / 4
        )
        total = len(read_current)
        ber.append(errors / total if total > 0 else 0)

    return np.array(ber)


def get_switching_probability(
    zero_currents: np.ndarray, one_currents: np.ndarray, threshold: float = None
) -> np.ndarray:
    """Calculate switching probability.

    Args:
        zero_currents: Array of zero state currents
        one_currents: Array of one state currents
        threshold: Switching threshold (if None, uses midpoint)

    Returns:
        Array of switching probabilities
    """
    if threshold is None:
        threshold = (np.mean(zero_currents) + np.mean(one_currents)) / 2

    switch_prob = []
    for zero_curr, one_curr in zip(zero_currents, one_currents):
        prob = abs(one_curr - zero_curr) / (abs(one_curr) + abs(zero_curr) + 1e-12)
        switch_prob.append(prob)

    return np.array(switch_prob)


def get_current_or_voltage(
    ltspice_data: ltspice.Ltspice, signal_name: str, case: int = 0
) -> np.ndarray:
    """Generic function to extract current or voltage signals.

    Args:
        ltspice_data: Parsed LTspice data object
        signal_name: Name of the signal to extract
        case: Case number for parametric sweeps

    Returns:
        Signal data array
    """
    return ltspice_data.get_data(signal_name, case=case)


def safe_max(arr: np.ndarray, mask: np.ndarray) -> float:
    """Safely compute maximum of masked array."""
    masked_arr = arr[mask]
    return np.max(masked_arr) if len(masked_arr) > 0 else 0.0


def safe_min(arr: np.ndarray, mask: np.ndarray) -> float:
    """Safely compute minimum of masked array."""
    masked_arr = arr[mask]
    return np.min(masked_arr) if len(masked_arr) > 0 else 0.0
