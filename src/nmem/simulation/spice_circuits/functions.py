import os
from typing import Tuple

import ltspice
import numpy as np
import scipy.io as sio

FILL_WIDTH = 5
VOUT_YMAX = 40
VOLTAGE_THRESHOLD = 2.0e-3


def get_persistent_current(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    signal_l = ltspice_data.get_data("Ix(HL:drain)", case=case)
    signal_r = ltspice_data.get_data("Ix(HR:drain)", case=case)
    return np.array([np.abs(signal_r[-1] - signal_l[-1]) / 2 * 1e6])


def get_write_current(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    signal = ltspice_data.get_data("I(R2)", case=case)
    return np.max(signal) * 1e6


def get_max_output(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    signal = ltspice_data.get_data("V(out)", case=case)
    return np.max(signal)


def get_processed_state_currents(data_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
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
    data_dict: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero_currents = []
    one_currents = []
    step_parameter = []
    for key, data in data_dict.items():
        step_parameter_str = get_step_parameter(data)
        zero_current, one_current = get_processed_state_currents(data)
        zero_currents.append(zero_current)
        one_currents.append(one_current)
        step_parameter.append(key)

    step_parameter = np.array(step_parameter)
    zero_currents = np.array(zero_currents)
    one_currents = np.array(one_currents)
    return step_parameter, zero_currents, one_currents, step_parameter_str


def process_data_dict_write_sweep(
    data_dict: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero_currents = []
    one_currents = []
    write_currents = []
    persistent_currents = []
    for key, data in data_dict.items():
        zero_current, one_current = get_processed_state_currents(data)
        zero_currents.append(zero_current)
        one_currents.append(one_current)
        write_currents.append(key)
        persistent_currents.append()
    return write_currents, zero_currents, one_currents


def get_write_sweep_data(file_path: str) -> dict:
    files = [f for f in os.listdir(file_path) if f.endswith(".csv")]
    files.sort()
    data_dict = {}
    for file in files:
        write_current = float(file[-15:-12])
        persistent_current = float(file[-9:-6])
        data = np.genfromtxt(file_path + file, delimiter=",")
        zero_current, one_current = get_processed_state_currents(data)
        read_margin = zero_current - one_current
        data_dict[write_current] = {
            "data": data,
            "zero_current": zero_current,
            "one_current": one_current,
            "write_current": write_current,
            "persistent_current": persistent_current,
            "read_margin": read_margin,
        }
    return data_dict


def get_bit_error_rate(
    read_zero_voltage: float,
    read_one_voltage: float,
    voltage_threshold: float = VOLTAGE_THRESHOLD,
) -> float:
    ber = np.where(
        (read_one_voltage < voltage_threshold)
        & (read_zero_voltage > voltage_threshold),
        1,
        0.5,
    )
    ber = np.where(
        (read_one_voltage > voltage_threshold)
        & (read_zero_voltage < voltage_threshold),
        0,
        ber,
    )
    return ber

def get_switching_probability(
    read_zero_voltage: float,
    read_one_voltage: float,
    voltage_threshold: float = VOLTAGE_THRESHOLD,
) -> float:
    switching_probability = np.where(
        (read_one_voltage > voltage_threshold)
        & (read_zero_voltage > voltage_threshold),
        1,
        0.5,
    )
    switching_probability = np.where(
        (read_one_voltage < voltage_threshold)
        & (read_zero_voltage < voltage_threshold),
        0,
        switching_probability,
    )
    return switching_probability

def get_current_or_voltage(
    ltsp: ltspice.Ltspice, signal: str, case: int = 0
) -> np.ndarray:
    signal_data = ltsp.get_data(f"I({signal})", case=case)
    if signal_data is None:
        signal_data = ltsp.get_data(f"V({signal})", case=case)
    return signal_data * 1e6

def safe_max(arr: np.ndarray, mask: np.ndarray) -> float:
    if np.any(mask):
        return np.max(arr[mask])
    return 0

def safe_min(arr: np.ndarray, mask: np.ndarray) -> float:
    if np.any(mask):
        return np.min(arr[mask])
    return 0

def process_read_data(ltsp: ltspice.Ltspice) -> dict:
    num_cases = ltsp.case_count

    read_current = np.zeros(num_cases)
    enable_read_current = np.zeros(num_cases)
    enable_write_current = np.zeros(num_cases)
    write_current = np.zeros(num_cases)
    persistent_current = np.zeros(num_cases)
    write_one_voltage = np.zeros(num_cases)
    write_zero_voltage = np.zeros(num_cases)
    read_zero_voltage = np.zeros(num_cases)
    read_one_voltage = np.zeros(num_cases)
    read_margin = np.zeros(num_cases)
    bit_error_rate = np.zeros(num_cases)
    switching_probability = np.zeros(num_cases)

    time_windows = {
        "persistent_current": (1.5e-7, 2e-7),
        "write_one": (1e-7, 1.5e-7),
        "write_zero": (5e-7, 5.5e-7),
        "read_one": (2e-7, 2.5e-7),
        "read_zero": (4e-7, 4.5e-7),
        "enable_write": (1e-7, 1.5e-7),
    }
    data_dict = {}
    for i in range(num_cases):
        time = ltsp.get_time(i)

        enable_current = ltsp.get_data("I(R1)", i) * 1e6
        channel_current = ltsp.get_data("I(R2)", i) * 1e6
        right_branch_current = ltsp.get_data("Ix(HR:drain)", i) * 1e6
        left_branch_current = ltsp.get_data("Ix(HL:drain)", i) * 1e6
        left_critical_current = get_current_or_voltage(ltsp, "ichl", i)
        right_critical_current = get_current_or_voltage(ltsp, "ichr", i)
        left_retrapping_current = get_current_or_voltage(ltsp, "irhl", i)
        right_retrapping_current = get_current_or_voltage(ltsp, "irhr", i)
        output_voltage = ltsp.get_data("V(out)", i)
        masks = {
            key: (time > start) & (time < end)
            for key, (start, end) in time_windows.items()
        }

        persistent_current[i] = safe_max(right_branch_current, masks["persistent_current"])
        write_current[i] = safe_max(channel_current, masks["write_one"])
        read_current[i] = safe_max(channel_current, masks["read_one"])
        enable_read_current[i] = safe_max(enable_current, masks["read_one"])
        enable_write_current[i] = safe_max(enable_current, masks["enable_write"])
        write_one_voltage[i] = safe_max(output_voltage, masks["write_one"])
        write_zero_voltage[i] = safe_min(output_voltage, masks["write_zero"])
        read_zero_voltage[i] = safe_max(output_voltage, masks["read_zero"])
        read_one_voltage[i] = safe_max(output_voltage, masks["read_one"])
        read_margin[i] = read_zero_voltage[i] - read_one_voltage[i]
        bit_error_rate[i] = get_bit_error_rate(
            read_zero_voltage[i], read_one_voltage[i]
        )
        switching_probability[i] = get_switching_probability(
            read_zero_voltage[i], read_one_voltage[i]
        )
        data_dict[i] = {
            "time": time,
            "tran_enable_current": enable_current,
            "tran_channel_current": channel_current,
            "tran_right_branch_current": right_branch_current,
            "tran_left_branch_current": left_branch_current,
            "tran_left_critical_current": left_critical_current,
            "tran_right_critical_current": right_critical_current,
            "tran_left_retrapping_current": left_retrapping_current,
            "tran_right_retrapping_current": right_retrapping_current,
            "tran_output_voltage": output_voltage,
            "write_current": write_current,
            "read_current": read_current,
            "enable_write_current": enable_write_current,
            "enable_read_current": enable_read_current,
            "read_zero_voltage": read_zero_voltage,
            "read_one_voltage": read_one_voltage,
            "write_one_voltage": write_one_voltage,
            "write_zero_voltage": write_zero_voltage,
            "persistent_current": persistent_current,
            "case_count": ltsp.case_count,
            "read_margin": read_margin,
            "bit_error_rate": bit_error_rate,
            "switching_probability": switching_probability,
        }
    return data_dict


def get_step_parameter(data_dict: dict) -> str:
    keys = [
        "write_current",
        "read_current",
        "enable_write_current",
        "enable_read_current",
    ]
    for key in keys:
        data = data_dict[key]
        if data[0] != data[1]:
            return key
    return None


def import_read_data(file_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    read_data = np.genfromtxt(file_path, delimiter=",")
    read_current = read_data[:, 0]
    read_output = read_data[:, 1:]
    return read_current, read_output


def import_csv_dir(file_path: str) -> dict:
    # get only csv from directory
    files = [f for f in os.listdir(file_path) if f.endswith(".csv")]
    files.sort()
    data_dict = {}
    for file in files:
        enable_read_current = float(file[-9:-6])
        data = np.genfromtxt(os.path.join(file_path, file), delimiter=",")
        data_dict[enable_read_current] = data

    return data_dict


def import_raw_dir(file_path: str) -> dict:
    # get only raw from directory
    files = [f for f in os.listdir(file_path) if f.endswith(".raw")]
    files.sort()
    data_dict = {}
    for file in files:
        data_dict[file] = ltspice.Ltspice(os.path.join(file_path, file)).parse()
    return data_dict


def save_enable_data_file(ltsp: ltspice.Ltspice):
    read_outputs = process_read_data(ltsp)
    write_current = read_outputs["write_current"]
    if len(write_current) > 1:
        write_current = write_current[0]
    enable_write_current = read_outputs["enable_write_current"]
    enable_write_current_array = enable_write_current.reshape(-1, 1)
    read_output = read_outputs["read_output"]

    data_dict = {
        "enable_write_current": enable_write_current,
        "read_output": read_output,
        "write_current": write_current,
        "persistent_current": read_outputs["persistent_current"],
        "write_zero_voltage": read_outputs["write_zero_voltage"],
        "write_one_voltage": read_outputs["write_one_voltage"],
    }

    sio.savemat(
        f"spice_simulation_raw/enable_write_sweep/enable_write_current_sweep_{write_current:03.0f}uA.mat",
        data_dict,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ltsp = ltspice.Ltspice("spice_simulation_raw/nmem_cell_write_200uA.raw").parse()
    data_dict = process_read_data(ltsp)
