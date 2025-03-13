import numpy as np
import ltspice
from typing import Tuple
import os 
import scipy.io as sio
from nmem.analysis.analysis import filter_first

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


def get_irhl(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(irhl)", case=case)[0] * 1e6])


def get_irhr(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(irhr)", case=case)[0] * 1e6])


def get_ichl(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(ichl)", case=case)[0] * 1e6])


def get_ichr(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    return np.array([ltspice_data.get_data("V(ichr)", case=case)[0] * 1e6])



def get_max_output(ltspice_data: ltspice.Ltspice, case: int = 0) -> np.ndarray:
    signal = ltspice_data.get_data("V(out)", case=case)
    return np.max(signal)



def process_data_dict(data_dict: dict) -> Tuple[float, float]:
    read_current = data_dict.get("read_current")
    read_output_0 = data_dict.get("read_output")[:, 0]
    read_output_1 = data_dict.get("read_output")[:, 1]

    zero_switch = np.argwhere(read_output_0 > VOLTAGE_THRESHOLD)
    one_switch = np.argwhere(read_output_1 > VOLTAGE_THRESHOLD)
    if len(zero_switch) == 0:
        zero_current = 0
    else:
        zero_current = read_current[zero_switch.flatten()[0]]
    if len(one_switch) == 0:
        one_current = 0
    else:
        one_current = read_current[one_switch.flatten()[0]]

    zero_current = float(zero_current)
    one_current = float(one_current)
    return zero_current, one_current


def process_data_array(array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if any(array[:, 1] > VOLTAGE_THRESHOLD):
        zero_switch = np.argwhere(array[:, 1] > VOLTAGE_THRESHOLD).flat[0]
    else:
        zero_switch = 0

    if any(array[:, 2] > VOLTAGE_THRESHOLD):
        one_switch = np.argwhere(array[:, 2] > VOLTAGE_THRESHOLD).flat[0]
    else:
        one_switch = 0
    zero_current = array[zero_switch, 0]
    one_current = array[one_switch, 0]
    return zero_current, one_current


def process_data_dict_sweep(data_dict: dict):
    zero_currents = []
    one_currents = []
    enable_read_currents = []
    for key, data in data_dict.items():
        if isinstance(data, np.ndarray):
            data_array = data
        else:
            data_array = data["data"]
        zero_current, one_current = process_data_array(data_array)
        zero_currents.append(zero_current)
        one_currents.append(one_current)
        enable_read_currents.append(key)

    enable_read_currents = np.array(enable_read_currents)
    zero_currents = np.array(zero_currents)
    one_currents = np.array(one_currents)
    return enable_read_currents, zero_currents, one_currents


def process_data_dict_write_sweep(data_dict: dict):
    zero_currents = []
    one_currents = []
    write_currents = []
    persistent_currents = []
    for key, data in data_dict.items():
        zero_current, one_current = process_data_array(data)
        zero_currents.append(zero_current)
        one_currents.append(one_current)
        write_currents.append(key)
        persistent_currents.append()
    return write_currents, zero_currents, one_currents


def get_read_margin(l: ltspice.Ltspice) -> float:
    read_outputs = process_read_data(l)
    zero_current, one_current = process_data_dict(read_outputs)
    read_margin = zero_current - one_current
    return float(read_margin)


def get_write_sweep_data(file_path: str):
    files = [f for f in os.listdir(file_path) if f.endswith(".csv")]
    files.sort()
    data_dict = {}
    for file in files:
        write_current = float(file[-15:-12])
        persistent_current = float(file[-9:-6])
        data = np.genfromtxt(file_path + file, delimiter=",")
        zero_current, one_current = process_data_array(data)
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



def process_read_data(l: ltspice.Ltspice):
    num_cases = l.case_count

    read_output = np.zeros((num_cases, 2))
    read_current = np.zeros(num_cases)
    enable_read_current = np.zeros(num_cases)
    enable_write_current = np.zeros(num_cases)
    write_current = np.zeros(num_cases)
    persistent_current = np.zeros(num_cases)
    write_one_voltage = np.zeros(num_cases)
    write_zero_voltage = np.zeros(num_cases)
    read_zero_voltage = np.zeros(num_cases)
    read_one_voltage = np.zeros(num_cases)

    time_windows = {
        "persistent_current": (1.5e-7, 2e-7),
        "write_one": (1e-7, 1.5e-7),
        "write_zero": (5e-7, 5.5e-7),
        "read_one": (2e-7, 2.5e-7),
        "read_zero": (6e-7, 6.8e-7),
        "enable_write": (1e-7, 1.5e-7),
    }

    for i in range(num_cases):
        time = l.get_time(i)
        enable_current = l.get_data("I(R1)", i) * 1e6
        channel_current = l.get_data("I(R2)", i) * 1e6
        branch_current = l.get_data("Ix(HR:drain)", i) * 1e6
        output_voltage = l.get_data("V(out)", i)

        masks = {
            key: (time > start) & (time < end)
            for key, (start, end) in time_windows.items()
        }

        read_output[i, 0] = np.max(output_voltage[masks["read_zero"]])
        read_output[i, 1] = np.max(output_voltage[masks["read_one"]])
        arr = branch_current[masks["persistent_current"]]
        persistent_current[i] = np.max(arr) if arr.size > 0 else 0
        write_current[i] = np.max(channel_current[masks["write_one"]])
        read_current[i] = np.max(channel_current[masks["read_one"]])
        enable_read_current[i] = np.max(enable_current[masks["read_one"]])
        enable_write_current[i] = np.max(enable_current[masks["enable_write"]])
        write_one_voltage[i] = np.max(output_voltage[masks["write_one"]])
        write_zero_voltage[i] = np.min(output_voltage[masks["write_zero"]])
        read_zero_voltage[i] = np.max(output_voltage[masks["read_zero"]])
        read_one_voltage[i] = np.max(output_voltage[masks["read_one"]])

    return {
        "read_current": read_current,
        "read_output": read_output,
        "enable_read_current": enable_read_current,
        "enable_write_current": enable_write_current,
        "write_current": write_current,
        "persistent_current": persistent_current,
        "write_one_voltage": write_one_voltage,
        "write_zero_voltage": write_zero_voltage,
        "read_zero": read_zero_voltage,
        "read_one": read_one_voltage,
    }


def import_read_data(file_path: str = None):
    read_data = np.genfromtxt(file_path, delimiter=",")
    read_current = read_data[:, 0]
    read_output = read_data[:, 1:]
    return read_current, read_output


def save_enable_data_file(l: ltspice.Ltspice):
    read_outputs = process_read_data(l)
    enable_read_current = read_outputs["enable_read_current"]
    read_current = read_outputs["read_current"]
    read_output = read_outputs["read_output"]
    np.savetxt(
        f"spice_simulation_raw/read_data_processed_{enable_read_current:.0f}uA.csv",
        np.hstack((read_current, read_output)),
        delimiter=",",
        header="read_current, read_output_0, read_output_1",
    )


def save_write_data_file(l: ltspice.Ltspice):
    data_dict = process_read_data(l)
    read_outputs = data_dict["read_output"]
    read_current = data_dict["read_current"]
    read_current = read_current.reshape(-1, 1)
    write_current = filter_first(data_dict["write_current"])
    read_output = data_dict["read_output"]
    persistent_current = filter_first(data_dict["persistent_current"])
    np.savetxt(
        f"read_data_write_current_sweep_{write_current:03.0f}uA_{persistent_current:03.0f}uA.csv",
        np.hstack((read_current, read_outputs)),
        delimiter=",",
        header="read_current, read_output_0, read_output_1",
    )


def save_enable_data_file(l: ltspice.Ltspice):
    read_outputs = process_read_data(l)
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


def import_csv_dir(file_path: str) -> dict:
    # get only csv from directory
    files = [f for f in os.listdir(file_path) if f.endswith(".csv")]
    files.sort()
    data_dict = {}
    for file in files:
        enable_read_current = float(file[-9:-6])
        data = np.genfromtxt(file_path + file, delimiter=",")
        data_dict[enable_read_current] = data

    return data_dict

