# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:01:31 2023

@author: omedeiro
"""

import collections.abc
import time
from datetime import datetime
from time import sleep
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import Axes
from qnnpy.functions.ntron import nTron
from scipy.optimize import curve_fit
from scipy.stats import norm
from tqdm import tqdm

from nmem.calculations.calculations import (
    calculate_critical_current,
    calculate_heater_power,
    htron_critical_current,
)
from nmem.analysis.analysis import get_fitting_points, plot_fit
from nmem.measurement.cells import CELLS, MITEQ_AMP_GAIN


def gauss(x: float, mu: float, sigma: float, A: float) -> float:
    return A * np.exp(-((x - mu) ** 2) / 2 / sigma**2)


def bimodal(
    x: float, mu1: float, sigma1: float, A1: float, mu2: float, sigma2: float, A2: float
) -> float:
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)


def bimodal_fit(
    x: np.ndarray,
    y: np.ndarray,
    expected: tuple,
    bounds: tuple = ((500, 2, 1e-6, 500, -30, 1e-6), (1200, 30, 1, 1200, 30, 1)),
) -> tuple:
    y = np.nan_to_num(y, posinf=0.0, neginf=0.0)
    params, cov = curve_fit(bimodal, x, y, expected, maxfev=5000, bounds=bounds)
    return params, cov


def read_sweep_scaled(
    measurement_settings: dict,
    current_cell: str,
    num_points: int = 15,
    start: float = 0.8,
    end: float = 0.95,
) -> dict:
    enable_read_current = measurement_settings.get("enable_read_current") * 1e6
    read_critical_current = (
        calculate_critical_current(enable_read_current, CELLS[current_cell]) * 1e-6
    )
    measurement_settings["y"] = np.linspace(
        read_critical_current * start, read_critical_current * end, num_points
    )
    return measurement_settings


def write_sweep_scaled(
    measurement_settings: dict,
    current_cell: str,
    num_points: int = 15,
    start: float = 0.8,
    end: float = 0.95,
) -> dict:
    write_current = measurement_settings.get("write_current") * 1e6
    write_critical_current = (
        calculate_critical_current(write_current, CELLS[current_cell]) * 1e-6
    )
    measurement_settings["y"] = np.linspace(
        write_critical_current * start, write_critical_current * end, num_points
    )
    return measurement_settings


def construct_currents(
    heater_currents: np.ndarray,
    slope: float,
    intercept: float,
    margin: float,
    max_critical_current: float,
) -> np.ndarray:
    bias_current_array = np.zeros((len(heater_currents), 2))
    for heater_current in heater_currents:
        critical_current = htron_critical_current(heater_current, slope, intercept)
        if critical_current > max_critical_current:
            critical_current = max_critical_current
        bias_currents = np.array(
            [
                critical_current * (1 - margin),
                critical_current * (1 + margin),
            ]
        )
        bias_current_array[heater_currents == heater_current] = bias_currents

    return bias_current_array


def filter_first(value):
    if isinstance(value, collections.abc.Iterable) and not isinstance(
        value, (str, bytes)
    ):
        return np.asarray(value).flatten()[0]
    return value


def get_param_mean(param: np.ndarray) -> np.ndarray:
    if round(param[2], 5) > round(param[5], 5):
        prm = param[0:2]
    else:
        prm = param[3:5]
    return prm


def reject_outliers(data: np.ndarray, m: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    ind = abs(data - np.mean(data)) < m * np.std(data)
    if len(ind[ind is False]) < 50:
        data = data[ind]
        print(f"Samples rejected {len(ind[ind is False])}")
        rejectInd = np.invert(ind)
    else:
        rejectInd = None
    return data, rejectInd


def update_dict(dict1: dict, dict2: dict) -> dict:
    result_dict = {}

    for key in dict1.keys():
        if isinstance(dict1[key], float) or isinstance(dict1[key], np.ndarray):
            try:
                result_dict[key] = np.dstack([dict1[key], dict2[key]])
            except Exception:
                print(f"could not stack {key}")
        else:
            result_dict[key] = dict1[key]
    return result_dict


def write_dict_to_file(file_path: str, save_dict: dict) -> None:
    with open(f"{file_path}_measurement_settings.txt", "w") as file:
        for key, value in save_dict.items():
            file.write(f"{key}: {value}\n")


def voltage2current(voltage: float, channel: int, measurement_settings: dict) -> float:
    cell: str = measurement_settings.get("cell")
    heater_dict: dict = measurement_settings.get("HEATERS")[int(cell[1])]
    spice_device_current: float = measurement_settings.get("spice_device_current")
    spice_input_voltage: float = heater_dict.get("spice_input_voltage")
    spice_heater_voltage: float = heater_dict.get("spice_heater_voltage")
    heater_resistance: float = measurement_settings["CELLS"][cell].get(
        "resistance_cryo"
    )
    if channel == 1:
        current = spice_device_current * (voltage / spice_input_voltage)
    if channel == 2:
        current = (
            voltage * (spice_heater_voltage / spice_input_voltage) / heater_resistance
        )
    return current


def current2voltage(current: float, channel: int, measurement_settings: dict) -> float:
    cell: str = measurement_settings.get("cell")
    heater_dict: dict = measurement_settings.get("HEATERS")[int(cell[1])]
    spice_device_current: float = measurement_settings.get("spice_device_current")
    spice_input_voltage: float = heater_dict.get("spice_input_voltage")
    spice_heater_voltage: float = heater_dict.get("spice_heater_voltage")
    heater_resistance: float = measurement_settings["CELLS"][cell].get(
        "resistance_cryo"
    )

    if channel == 1:
        voltage = spice_input_voltage * (current / spice_device_current)
    if channel == 2:
        voltage = (
            current * heater_resistance / (spice_heater_voltage / spice_input_voltage)
        )
    return voltage


def calculate_voltage(measurement_settings: dict) -> dict:
    enable_write_current = measurement_settings.get("enable_write_current")
    write_current = measurement_settings.get("write_current")
    read_current = measurement_settings.get("read_current")
    enable_read_current = measurement_settings.get("enable_read_current")

    enable_peak_current = max(enable_write_current, enable_read_current)
    enable_voltage = current2voltage(enable_peak_current, 2, measurement_settings)
    channel_voltage = current2voltage(
        read_current + write_current, 1, measurement_settings
    )
    channel_voltage_read = current2voltage(read_current, 1, measurement_settings)

    if read_current == 0:
        wr_ratio = 0
    else:
        wr_ratio = write_current / read_current

    if enable_read_current == 0:
        ewr_ratio = 0
    else:
        ewr_ratio = enable_write_current / enable_read_current

    measurement_settings.update(
        {
            "channel_voltage": channel_voltage,
            "channel_voltage_read": channel_voltage_read,
            "enable_voltage": enable_voltage,
            "wr_ratio": wr_ratio,
            "ewr_ratio": ewr_ratio,
        }
    )
    return measurement_settings


def calculate_threshold(read_zero_top: np.ndarray, read_one_top: np.ndarray) -> float:
    # Find the difference between the highest and lowest values in the read top arrays
    read_one_top_max = read_one_top.max()
    read_one_top_min = read_one_top.min()
    read_zero_top_max = read_zero_top.max()
    read_zero_top_min = read_zero_top.min()

    max_total = max(read_one_top_max, read_zero_top_max)
    min_total = min(read_one_top_min, read_zero_top_min)
    threshold = (max_total + min_total) / 2

    return threshold


def calculate_currents(
    time_zero: np.ndarray,
    time_one: np.ndarray,
    measurement_settings: dict,
    total_points: int,
    scope_timespan: float,
):
    num_meas = measurement_settings["num_meas"]
    read_current = measurement_settings["read_current"]

    time_one = time_one[1][0:num_meas]
    time_zero = time_zero[1][0:num_meas]

    time_one = time_one.flatten()
    time_zero = time_zero.flatten()

    if len(time_zero) < num_meas:
        time_zero.resize(num_meas, refcheck=False)
    if len(time_one) < num_meas:
        time_one.resize(num_meas, refcheck=False)

    read_time = (measurement_settings["read_width"] / total_points) * scope_timespan

    current_zero = time_zero / read_time * read_current
    current_one = time_one / read_time * read_current

    mean0, std0 = norm.fit(current_zero * 1e6)
    mean1, std1 = norm.fit(current_one * 1e6)

    distance = mean0 - mean1  # in microamps

    if len(current_zero) != 0 and len(current_one) != 0:
        x = np.linspace(mean0, mean1, 100)

        y0 = norm.pdf(x, mean0, std0)

        y1 = norm.pdf(x, mean1, std1)

        ydiff = np.subtract(y0, y1)

    return time_zero, time_one, current_zero, current_one, distance, x, y0, y1


def calculate_error_rate(t0: np.ndarray, t1: np.ndarray, num_meas: int):
    w0r1 = len(np.argwhere(t0 > 0))
    w1r0 = num_meas - len(np.argwhere(t1 > 0))

    ber = (w0r1 + w1r0) / (2 * num_meas)
    return ber, w0r1, w1r0


def calculate_bit_error_rate(
    write_1_read_0_errors: int, write_0_read_1_errors: int, num_meas: int
) -> float:
    bit_error_rate = float(
        (write_0_read_1_errors + write_1_read_0_errors) / (2 * num_meas)
    )
    return bit_error_rate


def calculate_power(data_dict: dict):
    one_voltage = data_dict.get("read_one_top")[1]
    zero_voltage = data_dict.get("read_zero_top")[1]
    read_current = data_dict.get("read_current")
    write_current = data_dict.get("write_current")

    write_voltage_high = np.mean(one_voltage)
    write_voltage_low = np.mean(zero_voltage)
    switch_voltage = write_voltage_high - write_voltage_low
    switch_voltage_preamp = switch_voltage / 10 ** (MITEQ_AMP_GAIN / 20)
    switch_power = switch_voltage_preamp * read_current
    switch_impedance = switch_voltage_preamp / read_current
    write_power = switch_impedance * write_current**2
    return switch_power, switch_impedance, write_power


def initilize_measurement(config: str, measurement_name: str) -> dict:
    b = nTron(config)

    b.inst.awg.write("SOURce1:FUNCtion:ARBitrary:FILTer OFF")
    b.inst.awg.write("SOURce2:FUNCtion:ARBitrary:FILTer OFF")

    current_cell = b.properties["Save File"]["cell"]
    sample_name = [
        b.sample_name,
        b.device_type,
        b.device_name,
        current_cell,
    ]
    sample_name = str("-".join(sample_name))
    date_str = time.strftime("%Y%m%d")
    time_str = time.strftime("%Y%m%d%H%M%S")
    measurement_name = f"{date_str}_{measurement_name}"

    measurement_settings = {
        "measurement_name": measurement_name,
        "sample_name": sample_name,
        "cell": current_cell,
        "time_str": time_str,
    }

    return measurement_settings, b


def initialize_data_dict(measurement_settings: dict) -> dict:
    scope_num_samples = measurement_settings.get("scope_num_samples")
    num_meas = measurement_settings.get("num_meas")
    sweep_x_len = len(measurement_settings.get("x"))
    sweep_y_len = len(measurement_settings.get("y"))

    data_dict = {
        "trace_chan_in": np.empty((2, scope_num_samples)),
        "trace_chan_out": np.empty((2, scope_num_samples)),
        "trace_enab": np.empty((2, scope_num_samples)),
        "read_zero_top": np.empty((1, num_meas)),
        "read_one_top": np.empty((1, num_meas)),
        "write_0_read_1": np.nan,
        "write_1_read_0": np.nan,
        "write_0_read_1_norm": np.nan,
        "write_1_read_0_norm": np.nan,
        "total_switches": np.nan,
        "total_switches_norm": np.nan,
        "channel_voltage": np.nan,
        "enable_voltage": np.nan,
        "bit_error_rate": np.nan,
        "sweep_x_len": sweep_x_len,
        "sweep_y_len": sweep_y_len,
    }
    return data_dict


def set_awg_on(b: nTron) -> None:
    b.inst.awg.set_output(True, 1)
    b.inst.awg.set_output(True, 2)


def set_awg_off(b: nTron) -> None:
    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)


def setup_scope_bert(
    b: nTron,
    measurement_settings: dict,
    division_zero: tuple = (4.5, 5.5),
    division_one: tuple = (9.5, 10.0),
):
    scope_horizontal_scale = measurement_settings.get("scope_horizontal_scale")
    scope_sample_rate = measurement_settings.get("scope_sample_rate")
    num_meas = measurement_settings.get("num_meas")

    b.inst.scope.set_horizontal_scale(
        scope_horizontal_scale, -scope_horizontal_scale * 5
    )
    b.inst.scope.set_sample_rate(max(scope_sample_rate, 1e6))

    b.inst.scope.set_measurement_gate("P3", division_zero[0], division_zero[1])
    b.inst.scope.set_measurement_gate("P4", division_one[0], division_one[1])

    b.inst.scope.set_math_trend_values("F5", num_meas * 2)
    b.inst.scope.set_math_trend_values("F6", num_meas * 2)
    b.inst.scope.set_math_vertical_scale("F5", 100e-3, 300e-3)
    b.inst.scope.set_math_vertical_scale("F6", 100e-3, 300e-3)


def setup_scope_8bit_bert(
    b: nTron,
    measurement_settings: dict,
):
    scope_horizontal_scale = measurement_settings["scope_horizontal_scale"]
    scope_sample_rate = measurement_settings["scope_sample_rate"]

    b.inst.scope.set_horizontal_scale(
        scope_horizontal_scale, -scope_horizontal_scale * 5
    )
    b.inst.scope.set_sample_rate(max(scope_sample_rate, 1e6))

    measurement_names = [f"P{i}" for i in range(1, 9)]
    for i, name in enumerate(measurement_names):
        b.inst.scope.set_measurement_gate(name, i, i + 0.2)


def setup_waveform(b: nTron, measurement_settings: dict):
    bitmsg_channel = measurement_settings.get("bitmsg_channel")
    bitmsg_enable = measurement_settings.get("bitmsg_enable")
    channel_voltage = measurement_settings.get("channel_voltage")
    enable_voltage = measurement_settings.get("enable_voltage")
    sample_rate = measurement_settings.get("sample_rate")
    ramp_read = measurement_settings.get("ramp_read", False)

    if enable_voltage > 300e-3:
        raise ValueError("enable voltage too high")

    if channel_voltage > 3.0:
        raise ValueError("channel voltage too high")

    if channel_voltage == 0:
        bitmsg_channel = "N" * len(bitmsg_channel)
    if enable_voltage == 0:
        bitmsg_enable = "N" * len(bitmsg_enable)

    write_sequence(b, bitmsg_channel, 1, measurement_settings, ramp_read=ramp_read)
    b.inst.awg.set_vpp(channel_voltage, 1)
    b.inst.awg.set_arb_sample_rate(sample_rate, 1)

    write_sequence(b, bitmsg_enable, 2, measurement_settings, ramp_read=ramp_read)
    b.inst.awg.set_vpp(enable_voltage, 2)
    b.inst.awg.set_arb_sample_rate(sample_rate, 2)

    b.inst.awg.set_arb_sync()
    sleep(.1)


def create_waveforms(
    num_samples: int = 256,
    width: int = 30,
    height: int = 1,
    phase: int = 0,
    ramp: bool = False,
) -> np.ndarray:
    """
    Create waveforms with specified parameters.

    Parameters:
        num_samples (int): The number of samples in the waveform. Default is 256.
        width (int): The width of the waveform. Default is 30.
        height (int): The height of the waveform. Default is 1.
        phase (int): The phase shift of the waveform. Default is 0.
        ramp (bool): Whether to create a ramp waveform. Default is False.

    Returns:
        numpy.ndarray: The generated waveform.

    """
    waveform = np.zeros(num_samples)

    middle = np.floor(num_samples / 2) + phase
    half = np.floor(width / 2)
    start = int(middle - half)
    stop = int(start + width)

    if start < 0:
        start = 0
    if stop > num_samples:
        stop = int(num_samples)

    if ramp:
        waveform[start:stop] = np.linspace(0, height, int(np.floor(width)))
    else:
        waveform[start:stop] = height

    return waveform


def create_waveforms_edge(
    num_samples: int = 256,
    width: int = 30,
    height: int = 1,
    phase: int = 0,
    edge: int = 5,
) -> np.ndarray:
    waveform = np.zeros(num_samples)
    middle = np.floor(num_samples / 2) + phase
    half = np.floor(width / 2)
    start = int(middle - half)
    stop = int(start + width)

    if start < 0:
        start = edge
    if stop > num_samples:
        stop = int(num_samples) - edge

    waveform[start:stop] = height
    waveform[start - edge : start] = np.linspace(0, height, edge)
    waveform[stop : stop + edge] = np.linspace(height, 0, edge)

    return waveform


def create_dataframe(data_dict: dict) -> pd.DataFrame:
    write_current = data_dict.get("write_current")
    read_current = data_dict.get("read_current")
    enable_write_current = data_dict.get("enable_write_current")
    enable_read_current = data_dict.get("enable_read_current")
    cell = data_dict.get("cell")
    max_heater_current = data_dict["CELLS"][cell].get("max_heater_current", 1.0)
    write_critical_current = calculate_critical_current(
        enable_write_current * 1e6, data_dict["CELLS"][cell]
    )
    read_critical_current = calculate_critical_current(
        enable_read_current * 1e6, data_dict["CELLS"][cell]
    )

    enable_write_power = calculate_heater_power(
        data_dict["enable_write_current"],
        data_dict["CELLS"][cell]["resistance_cryo"],
    )
    enable_read_power = calculate_heater_power(
        data_dict["enable_read_current"],
        data_dict["CELLS"][cell]["resistance_cryo"],
    )
    switch_power, switch_impedance, write_power = calculate_power(data_dict)

    param_dict = {
        "Write Current [uA]": [write_current * 1e6],
        "Read Current [uA]": [read_current * 1e6],
        "Enable Write Current [uA]": [enable_write_current * 1e6],
        "Enable Read Current [uA]": [enable_read_current * 1e6],
        "Write Critical Current [uA]": [write_critical_current],
        "Write Enable Fraction": [enable_write_current * 1e6 / max_heater_current],
        "Read Critical Current [uA]": [read_critical_current],
        "Read Enable Fraction": [enable_read_current * 1e6 / max_heater_current],
        "Max Heater Current [uA]": [max_heater_current],
        "Enable Write Power [uW]": [enable_write_power * 1e6],
        "Enable Read Power [uW]": [enable_read_power * 1e6],
        "Switch Power [uW]": [switch_power * 1e6],
        "Switch Impedance [Ohm]": [switch_impedance],
        "Write Power [uW]": [write_power * 1e6],
        "Write / Read Ratio": [data_dict["wr_ratio"]],
        "Enable Write / Read Ratio": [data_dict["ewr_ratio"]],
        "Write 0 Read 1": [data_dict["write_0_read_1"]],
        "Write 1 Read 0": [data_dict["write_1_read_0"]],
        "Bit Error Rate": [data_dict["bit_error_rate"]],
    }
    if write_current > 0:
        param_dict.update(
            {
                "Write Bias Fraction": [
                    write_current / (write_critical_current * 1e-6)
                ],
            }
        )

    if read_current > 0:
        param_dict.update(
            {
                "Read Bias Fraction": [read_current / (read_critical_current * 1e-6)],
            }
        )
    pd.set_option("display.float_format", "{:.3f}".format)
    param_df = pd.DataFrame(param_dict.values(), index=param_dict.keys())
    param_df.columns = ["Value"]
    return param_df


def write_waveforms(b: nTron, write_string: str, chan: int):
    name = "CHAN"
    sequence = f'"{name}",{write_string}'
    n = str(len(sequence))
    header = f"SOUR{chan}:DATA:SEQ #{len(n)}{n}"
    message = header + sequence
    # clears volatile waveform memory

    b.inst.awg.pyvisa.write_raw(message)
    b.inst.awg.write(f"SOURce{chan}:FUNCtion:ARBitrary {name}")
    # b.inst.awg.get_error()

    b.inst.awg.set_arb_sync()


RISING_EDGE = 10


def load_waveforms(
    b: nTron,
    measurement_settings: dict,
    chan: int = 1,
    ramp_read: bool = True,
):
    ww = measurement_settings.get("write_width")
    rw = measurement_settings.get("read_width")

    wr_ratio = measurement_settings.get("write_current") / measurement_settings.get(
        "read_current"
    )

    eww = measurement_settings.get("enable_write_width")
    erw = measurement_settings.get("enable_read_width")

    ew_phase = measurement_settings.get("enable_write_phase")
    er_phase = measurement_settings.get("enable_read_phase")

    ewr_ratio = measurement_settings.get(
        "enable_write_current"
    ) / measurement_settings.get("enable_read_current")

    num_points = measurement_settings.get("num_points", 256)

    if wr_ratio >= 1:
        write_0 = create_waveforms_edge(width=ww, height=-1, edge=1)
        write_1 = create_waveforms_edge(width=ww, height=1, edge=1)
        read_wave = create_waveforms_edge(
            width=rw, height=1 / wr_ratio, edge=RISING_EDGE
        )
    else:
        write_0 = create_waveforms_edge(width=ww, height=-wr_ratio, edge=RISING_EDGE)
        write_1 = create_waveforms_edge(width=ww, height=wr_ratio, edge=RISING_EDGE)
        read_wave = create_waveforms_edge(width=rw, height=1, edge=RISING_EDGE)

    if ewr_ratio >= 1:
        enable_write = create_waveforms_edge(
            width=eww, height=1, phase=ew_phase, edge=1
        )
        enable_read = create_waveforms_edge(
            width=erw, height=1 / ewr_ratio, phase=er_phase, edge=1
        )
    else:
        enable_write = create_waveforms_edge(
            width=eww, height=ewr_ratio, phase=ew_phase, edge=1
        )
        enable_read = create_waveforms_edge(width=erw, height=1, phase=er_phase, edge=1)

    null_wave = create_waveforms(height=0)

    b.inst.awg.set_arb_wf(write_0, name="WRITE0", num_samples=num_points, chan=chan)
    b.inst.awg.set_arb_wf(write_1, name="WRITE1", num_samples=num_points, chan=chan)
    b.inst.awg.set_arb_wf(read_wave, name="READ", num_samples=num_points, chan=chan)
    b.inst.awg.set_arb_wf(enable_write, name="ENABW", num_samples=num_points, chan=chan)
    b.inst.awg.set_arb_wf(enable_read, name="ENABR", num_samples=num_points, chan=chan)
    b.inst.awg.set_arb_wf(null_wave, name="WNULL", num_samples=num_points, chan=chan)

    b.inst.awg.write(f"SOURce{chan}:DATA:VOLatile:CLEar")

    write1 = '"INT:\WRITE1.ARB"'
    write0 = '"INT:\WRITE0.ARB"'
    read = '"INT:\READ.ARB"'
    enabw = '"INT:\ENABW.ARB"'
    enabr = '"INT:\ENABR.ARB"'
    wnull = '"INT:\WNULL.ARB"'

    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {write0}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {write1}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {read}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {enabw}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {enabr}")
    b.inst.awg.write(f"MMEM:LOAD:DATA{chan} {wnull}")


def write_sequence(
    b: nTron,
    message: str,
    channel: int,
    measurement_settings: dict,
    ramp_read: bool = True,
):
    write1 = '"INT:\WRITE1.ARB"'
    write0 = '"INT:\WRITE0.ARB"'
    read = '"INT:\READ.ARB"'
    enabw = '"INT:\ENABW.ARB"'
    enabr = '"INT:\ENABR.ARB"'
    wnull = '"INT:\WNULL.ARB"'

    write_string = []
    start_flag = True
    for c in message:
        if start_flag is True:
            suffix = ",0,once,highAtStartGoLow,50"
        else:
            suffix = ",0,once,maintain,50"

        if c == "0":
            write_string.append(write0 + suffix)
        if c == "1":
            write_string.append(write1 + suffix)
        if c == "W":
            write_string.append(enabw + suffix)
        if c == "N":
            write_string.append(wnull + suffix)
        if c == "E":
            write_string.append(enabr + suffix)
        if c == "R":
            write_string.append(read + suffix)

        start_flag = False

    write_string = ",".join(write_string)
    # print(write_string)

    load_waveforms(b, measurement_settings, chan=channel, ramp_read=ramp_read)
    write_waveforms(b, write_string, channel)


def replace_bit(bitmsg: str, i: int, bit: str) -> str:
    return bitmsg[:i] + bit + bitmsg[i + 1 :]


def get_traces(b: nTron, scope_samples: int = 5000) -> dict:
    sleep(1)
    b.inst.scope.set_trigger_mode("Single")
    sleep(0.1)
    trace_chan_in: np.ndarray = b.inst.scope.get_wf_data("C1")
    sleep(0.1)
    trace_chan_out: np.ndarray = b.inst.scope.get_wf_data("C2")
    sleep(0.1)
    trace_enab: np.ndarray = b.inst.scope.get_wf_data("C4")
    sleep(0.1)
    trace_write_avg: np.ndarray = b.inst.scope.get_wf_data("F1")
    sleep(0.1)
    trace_ewrite_avg: np.ndarray = b.inst.scope.get_wf_data("F2")
    sleep(0.1)
    trace_eread_avg: np.ndarray = b.inst.scope.get_wf_data("F3")
    sleep(0.1)
    trace_read0_avg: np.ndarray = b.inst.scope.get_wf_data("F7")
    sleep(0.1)
    trace_read1_avg: np.ndarray = b.inst.scope.get_wf_data("F4")
    sleep(0.1)
    trace_read0: np.ndarray = b.inst.scope.get_wf_data("Z5")
    sleep(0.1)
    trace_read1: np.ndarray = b.inst.scope.get_wf_data("Z4")
    trace_chan_in = np.resize(trace_chan_in, (2, scope_samples))
    trace_chan_out = np.resize(trace_chan_out, (2, scope_samples))
    trace_enab = np.resize(trace_enab, (2, scope_samples))

    b.inst.scope.set_trigger_mode("Normal")
    sleep(0.1)

    trace_dict: dict = {
        "trace_chan_in": trace_chan_in,
        "trace_chan_out": trace_chan_out,
        "trace_enab": trace_enab,
        "trace_write_avg": trace_write_avg,
        "trace_ewrite_avg": trace_ewrite_avg,
        "trace_eread_avg": trace_eread_avg,
        "trace_read0_avg": trace_read0_avg,
        "trace_read1_avg": trace_read1_avg,
        "trace_read0": trace_read0,
        "trace_read1": trace_read1,
    }

    return trace_dict


def get_traces_sequence(b: nTron, num_meas: int = 100, num_samples: int = 5000):
    data_dict = []
    for c in ["C1", "C2", "C4", "F4"]:
        x, y = b.inst.scope.get_wf_data(channel=c)
        interval = abs(x[0] - x[1])
        xlist = []
        ylist = []
        totdp = np.int64(np.size(x) / num_meas)

        for j in range(1):
            xx = x[0 + j * totdp : totdp + j * totdp] - totdp * interval * j
            yy = y[0 + j * totdp : totdp + j * totdp]

            xlist.append(xx[0 : int(num_samples)])
            ylist.append(yy[0 : int(num_samples)])

        trace_dict = [xlist, ylist]
        data_dict.append(trace_dict)

    data0 = data_dict[0]
    data1 = data_dict[1]
    data2 = data_dict[2]
    data3 = data_dict[3]

    data0 = np.resize(data0, (2, num_samples))
    data1 = np.resize(data1, (2, num_samples))
    data2 = np.resize(data2, (2, num_samples))
    data3 = np.resize(data3, (2, num_samples))

    try:
        time_est = round(b.inst.scope.get_parameter_value("P2"), 8)
        b.inst.scope.set_math_vertical_scale("F1", 50e-9, time_est)
        b.inst.scope.set_math_vertical_scale("F2", 50e-9, time_est)
    except Exception:
        sleep(1e-4)

    return data0, data1, data2, data3


def plot_message(ax: plt.Axes, message: str):
    message_dict = {
        "0": "Write 0",
        "1": "Write 1",
        "W": "Enable\nWrite",
        "E": "Enable\nRead",
        "R": "Read",
        "N": "-",
    }
    for i, bit in enumerate(message):
        ax.text(
            i + 0.5,
            ax.get_ylim()[1] * 0.85,
            message_dict[bit],
            ha="center",
            va="center",
        )

    return ax





def plot_ber(ax: Axes, x: np.ndarray, y: np.ndarray, ber: np.ndarray) -> Axes:
    dx = x[1] - x[0]
    xstart = x[0]
    xstop = x[-1]
    dy = y[1] - y[0]
    ystart = y[0]
    ystop = y[-1]

    cmap = plt.get_cmap("RdBu", 100).reversed()
    ax.imshow(
        np.reshape(ber, (len(x), len(y))),
        extent=[
            (-0.5 * dx + xstart),
            (0.5 * dx + xstop),
            (-0.5 * dy + ystart),
            (0.5 * dy + ystop),
        ],
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )
    return ax


def plot_header(fig: plt.Figure, data_dict: dict) -> plt.Figure:
    sample_name: str = filter_first(data_dict.get("sample_name"))
    measurement_name: str = filter_first(data_dict.get("measurement_name"))
    time_str: str = filter_first(data_dict.get("time_str"))
    channel_voltage: float = filter_first(data_dict.get("channel_voltage"))
    enable_voltage: float = filter_first(data_dict.get("enable_voltage"))
    write_current: float = filter_first(data_dict.get("write_current"))
    read_current: float = filter_first(data_dict.get("read_current"))
    enable_write_current: float = filter_first(data_dict.get("enable_write_current"))
    enable_read_current: float = filter_first(data_dict.get("enable_read_current"))
    bitmsg_channel: str = filter_first(data_dict.get("bitmsg_channel"))
    bitmsg_enable: str = filter_first(data_dict.get("bitmsg_enable"))
    write_width: int = filter_first(data_dict.get("write_width"))
    read_width: int = filter_first(data_dict.get("read_width"))
    enable_write_width: int = filter_first(data_dict.get("enable_write_width"))
    enable_read_width: int = filter_first(data_dict.get("enable_read_width"))
    enable_write_phase: int = filter_first(data_dict.get("enable_write_phase"))
    enable_read_phase: int = filter_first(data_dict.get("enable_read_phase"))

    fig.suptitle(
        (
            f"{sample_name} -- {measurement_name} -- {time_str}\n"
            f"Vcpp: {channel_voltage * 1e3:.1f} mV, Vepp: {enable_voltage * 1e3:.1f} mV\n"
            f"Write Current: {write_current * 1e6:.1f} uA, Read Current: {read_current * 1e6:.0f} uA\n"
            f"Enable Write Current: {enable_write_current * 1e6:.1f} uA, Enable Read Current: {enable_read_current * 1e6:.1f} uA\n"
            f"Channel Message: {bitmsg_channel}, Channel Enable: {bitmsg_enable}\n"
            f"Write Width: {write_width}, Read Width: {read_width}, \n"
            f"Enable Write Width: {enable_write_width}, Enable Read Width: {enable_read_width}, "
            f"Enable Write Phase: {enable_write_phase}, Enable Read Phase: {enable_read_phase}"
        )
    )
    fig.subplots_adjust(top=0.85)
    return fig


def plot_waveform(ax: Axes, waveform: np.ndarray, **kwargs) -> Axes:
    ax.plot(waveform[0] * 1e6, waveform[1] * 1e3, **kwargs)
    ax.legend(loc=4)
    ax.set_ylabel("voltage (mV)")
    ax.set_xlabel("time (us)")
    ax.set_ymargin(0.3)
    return ax


def plot_trend(ax: Axes, trend: np.ndarray, **kwargs) -> Axes:
    ax.plot(trend, ls="none", marker="o", **kwargs)
    ax.set_ylabel("read voltage (V)")
    ax.set_xlabel("sample")
    ax.legend(loc=4)
    ax.set_ylim([0, 0.65])
    return ax


def plot_waveforms_bert(
    axs: List[Axes],
    data_dict: dict,
) -> List[Axes]:
    trace_chan_in = data_dict.get("trace_chan_in")
    trace_chan_out = data_dict.get("trace_chan_out")
    trace_enab = data_dict.get("trace_enab")
    read_zero_top = data_dict.get("read_zero_top")
    read_one_top = data_dict.get("read_one_top")

    scope_timespan = data_dict.get("scope_timespan")
    threshold = data_dict.get("voltage_threshold")
    bitmsg_channel = data_dict.get("bitmsg_channel")
    bitmsg_enable = data_dict.get("bitmsg_enable")

    numpoints = int((len(trace_chan_in[1]) - 1) / 2)
    cmap = plt.cm.viridis(np.linspace(0, 1, 200))
    C1 = 45
    C2 = 135

    waveform_dict = {
        0: {
            "waveform": trace_chan_in,
            "label": "input signal",
            "color": cmap[C1, :],
            "bit_message": bitmsg_channel,
        },
        1: {
            "waveform": trace_chan_out,
            "label": "channel",
            "color": cmap[C1, :],
            "bit_message": bitmsg_channel,
        },
        2: {
            "waveform": trace_enab,
            "label": "enable",
            "color": cmap[C1, :],
            "bit_message": bitmsg_enable,
        },
    }
    for key in waveform_dict.keys():
        plot_waveform(
            axs[key],
            waveform_dict[key].get("waveform"),
            label=waveform_dict[key].get("label"),
            color=waveform_dict[key].get("color"),
        )
        plot_message(axs[key], waveform_dict[key].get("bit_message"))

    plot_trend(axs[3], read_zero_top, label="READ 0", color=cmap[C1, :])
    plot_trend(axs[3], read_one_top, label="READ 1", color=cmap[C2, :])
    axs[3].hlines(threshold, 0, len(read_zero_top), color="r", label="threshold")

    fig = plt.gcf()

    plot_header(fig, data_dict)

    fig.tight_layout()

    return axs


def plot_parameter(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    **kwargs,
):
    ax.plot(x, y, marker="o", **kwargs)

    return ax


def get_extent(x: np.ndarray, y: np.ndarray, zarray: np.ndarray) -> List[float]:
    dx = x[1] - x[0]
    xstart = x[0]
    xstop = x[-1]
    dy = y[1] - y[0]
    ystart = y[0]
    ystop = y[-1]
    return [
        (-0.5 * dx + xstart),
        (0.5 * dx + xstop),
        (-0.5 * dy + ystart),
        (0.5 * dy + ystop),
    ]


def plot_array(
    ax: Axes,
    data_dict: dict,
    c_name: str,
    cmap=None,
    norm=False,
):
    x_name: str = data_dict.get("sweep_parameter_x")
    y_name: str = data_dict.get("sweep_parameter_y")
    x, y, zarray = build_array(data_dict, c_name)
    zextent = get_extent(x, y, zarray)

    if not cmap:
        cmap = plt.get_cmap("RdBu", 100).reversed()

    if norm:
        cmax = 1
        cmin = 0
    else:
        cmax = np.nanmax(zarray)
        cmin = 0

    ax.matshow(
        zarray,
        extent=zextent,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
    )
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    xfit, yfit = get_fitting_points(x, y, zarray)
    ax.plot(xfit, yfit, "ro")
    plot_fit(ax, xfit, yfit)
    return ax


def plot_ber_sweep(
    ax: Axes,
    data_dict: dict,
    parameter_z: str,
) -> Axes:

    plot_array(ax, data_dict, parameter_z)

    return ax


def run_bitwise(b: nTron, measurement_settings: dict):
    b.inst.scope.set_trigger_mode("Stop")
    sleep(0.1)
    b.inst.scope.clear_sweeps()
    b.inst.scope.set_trigger_mode("Single")

    t1 = b.inst.scope.get_parameter_value("P1")
    t2 = b.inst.scope.get_parameter_value("P2")
    t3 = b.inst.scope.get_parameter_value("P3")
    t4 = b.inst.scope.get_parameter_value("P4")
    t5 = b.inst.scope.get_parameter_value("P5")
    t6 = b.inst.scope.get_parameter_value("P6")
    t7 = b.inst.scope.get_parameter_value("P7")
    t8 = b.inst.scope.get_parameter_value("P8")

    bits = [t1, t2, t3, t4, t5, t6, t7, t8]
    return bits


def run_realtime_bert(
    b: nTron, measurement_settings: dict, channel: str = "F5"
) -> dict:
    num_meas = measurement_settings.get("num_meas")
    threshold = measurement_settings.get("voltage_threshold")

    b.inst.scope.set_trigger_mode("Normal")
    sleep(0.5)
    b.inst.scope.clear_sweeps()
    sleep(0.1)
    with tqdm(total=num_meas, leave=False) as pbar:
        while b.inst.scope.get_num_sweeps(channel) < num_meas:
            sleep(0.1)
            n = b.inst.scope.get_num_sweeps(channel)
            pbar.update(n - pbar.n)

    b.inst.scope.set_trigger_mode("Stop")

    result_dict = get_results(b, num_meas, threshold)

    return result_dict


def get_results(b: nTron, num_meas: int, threshold: float) -> dict:
    read_zero_top, read_one_top = get_trend(b, num_meas)

    # READ 1: above threshold (voltage)
    write_0_read_1 = np.array([(read_zero_top > threshold).sum()])

    # READ 0: below threshold (no voltage)
    write_1_read_0 = np.array([(read_one_top < threshold).sum()])

    write_0_read_1_norm = write_0_read_1 / (num_meas * 2)
    write_1_read_0_norm = write_1_read_0 / (num_meas * 2)
    bit_error_rate = calculate_bit_error_rate(write_1_read_0, write_0_read_1, num_meas)

    total_switches = write_0_read_1 + (num_meas - write_1_read_0)
    total_switches_norm = total_switches / (num_meas * 2)
    result_dict = {
        "write_0_read_1": write_0_read_1,
        "write_1_read_0": write_1_read_0,
        "write_0_read_1_norm": write_0_read_1_norm,
        "write_1_read_0_norm": write_1_read_0_norm,
        "read_zero_top": read_zero_top,
        "read_one_top": read_one_top,
        "bit_error_rate": bit_error_rate,
        "total_switches": total_switches,
        "total_switches_norm": total_switches_norm,
    }
    return result_dict


def get_trend(b: nTron, num_meas: int) -> Tuple[np.ndarray, np.ndarray]:
    read_zero_top = b.inst.scope.get_wf_data("F5")
    read_one_top = b.inst.scope.get_wf_data("F6")

    read_zero_top = read_zero_top[1][0:num_meas]
    read_one_top = read_one_top[1][0:num_meas]

    read_zero_top = read_zero_top.flatten()
    read_one_top = read_one_top.flatten()

    if len(read_zero_top) < num_meas:
        read_zero_top.resize(num_meas, refcheck=False)
    if len(read_one_top) < num_meas:
        read_one_top.resize(num_meas, refcheck=False)

    return read_zero_top, read_one_top


def run_sequence_bert(b: nTron, measurement_settings: dict):
    num_meas = measurement_settings.get("num_meas")
    threshold = measurement_settings.get("voltage_threshold")

    b.inst.scope.clear_sweeps()
    sleep(0.1)
    b.inst.scope.set_trigger_mode("Single")
    sleep(1)
    while b.inst.scope.get_trigger_mode() != "Stopped\n":
        sleep(1)
        print("sampling...")

    result_dict = get_results(b, num_meas, threshold)

    return result_dict


def run_sequence(b: nTron, num_meas: int = 100, num_samples: int = 5e3):
    b.inst.scope.clear_sweeps()
    sleep(0.1)
    b.inst.scope.set_trigger_mode("Single")
    sleep(1)
    while b.inst.scope.get_trigger_mode() != "Stopped\n":
        sleep(1)
        print("sampling...")

    t1 = b.inst.scope.get_wf_data("F1")
    t0 = b.inst.scope.get_wf_data("F2")

    save_full_dict = 0
    if save_full_dict == 1:
        full_dict = {}
        for c in ["C1", "C2", "C3", "C4"]:
            x, y = b.inst.scope.get_wf_data(channel=c)
            interval = abs(x[0] - x[1])
            xlist = []
            ylist = []
            totdp = np.int64(np.size(x) / num_meas)

            for j in range(num_meas):
                xx = x[0 + j * totdp : totdp + j * totdp] - totdp * interval * j
                yy = y[0 + j * totdp : totdp + j * totdp]

                xlist.append(xx[0 : int(num_samples)])
                ylist.append(yy[0 : int(num_samples)])

            data_dict = {c + "x": xlist, c + "y": ylist}
            full_dict.update(data_dict)
    else:
        full_dict = {}

    return t0, t1, full_dict


def run_sequence_v1(b: nTron, num_meas: int = 100):
    b.inst.scope.set_trigger_mode("Stop")
    sleep(0.1)

    sleep(0.1)
    b.inst.scope.set_segments(50)
    sleep(0.5)
    b.inst.scope.set_trigger_mode("Normal")
    sleep(1)
    b.inst.scope.clear_sweeps()
    while b.inst.scope.get_num_sweeps() < num_meas:
        sleep(0.1)
        n = b.inst.scope.get_num_sweeps()
        print(f"sampling...{n} / {num_meas}")

    t0 = b.inst.scope.get_wf_data("F1")

    t1 = b.inst.scope.get_wf_data("F2")

    return t0, t1


def run_bitwise_measurement(b: nTron, measurement_settings: dict):
    measurement_settings = calculate_voltage(measurement_settings)

    setup_waveform(b, measurement_settings)

    b.inst.awg.set_output(True, 1)
    b.inst.awg.set_output(True, 2)

    b.inst.scope.clear_sweeps()

    bits = run_bitwise(b, measurement_settings)

    print(bits)
    set_awg_off(b)

    # bit_error_rate = calculate_bit_error_rate(
    #     meas_dict["write_1_read_0"], meas_dict["write_0_read_1"], num_meas
    # )
    # DATA_DICT = {
    #     **meas_dict,
    #     "trace_chan_in": trace_chan_in,
    #     "trace_chan_out": trace_chan_out,
    #     "trace_enab": trace_enab,
    #     "bit_error_rate": bit_error_rate,
    # }
    # print(f"Bit Error Rate: {bit_error_rate:.2e}")
    # if plot:
    #     plot_waveforms_bert(DATA_DICT, measurement_settings)

    # return DATA_DICT


def run_measurement(
    b: nTron,
    measurement_settings: dict,
    plot: bool = False,
) -> dict:
    measurement_settings = calculate_voltage(measurement_settings)
    scope_samples: int = int(measurement_settings.get("scope_num_samples"))

    ######################################################

    setup_waveform(b, measurement_settings)

    b.inst.awg.set_output(True, 1)
    b.inst.awg.set_output(True, 2)

    b.inst.scope.clear_sweeps()

    data_dict = run_realtime_bert(b, measurement_settings)
    data_dict.update(get_traces(b, scope_samples))
    data_dict.update(measurement_settings)

    set_awg_off(b)

    if plot:
        fig, axs = plt.subplots(4, 1, figsize=(10, 10))
        plot_waveforms_bert(axs, data_dict)
        plt.show()

    return data_dict


def run_sweep(
    b: nTron,
    measurement_settings: dict,
    sweep_parameter_x: str,
    sweep_parameter_y: str,
    plot_measurement=False,
    division_zero: Tuple[float, float] = (4.5, 5.5),
    division_one: Tuple[float, float] = (9.5, 10),
) -> dict:
    save_dict = {}

    setup_scope_bert(
        b, measurement_settings, division_zero=division_zero, division_one=division_one
    )

    for x in measurement_settings["x"]:
        for y in measurement_settings["y"]:
            measurement_settings.update({sweep_parameter_x: x})
            measurement_settings.update({sweep_parameter_y: y})

            data_dict = initialize_data_dict(measurement_settings)
            data_dict.update(
                run_measurement(
                    b,
                    measurement_settings,
                    plot=plot_measurement,
                    ramp_read=False,
                )
            )

            data_dict.update(measurement_settings)

            param_dataframe = create_dataframe(data_dict)

            if len(save_dict.items()) == 0:
                save_dict = data_dict
            else:
                save_dict = update_dict(save_dict, data_dict)

    return save_dict


def run_sweep_subset(
    b: nTron,
    measurement_settings: dict,
    plot_measurement: bool = False,
    division_zero: Tuple[float, float] = (4.5, 5.5),
    division_one: Tuple[float, float] = (9.5, 10),
) -> dict:
    save_dict = {}
    sweep_parameter_x: str = measurement_settings.get("sweep_parameter_x")
    sweep_parameter_y: str = measurement_settings.get("sweep_parameter_y")
    xarray: np.ndarray = measurement_settings.get("x")
    yarray: np.ndarray = measurement_settings.get("y")

    setup_scope_bert(
        b, measurement_settings, division_zero=division_zero, division_one=division_one
    )
    for idx, x in enumerate(xarray):
        for _, y in enumerate(yarray):

            measurement_settings.update({sweep_parameter_x: x})
            measurement_settings.update({sweep_parameter_y: y})

            measurement_settings = calculate_voltage(measurement_settings)

            data_dict = initialize_data_dict(measurement_settings)
            if (
                y > measurement_settings["y_subset"][idx][0]
                and y < measurement_settings["y_subset"][idx][-1]
            ):
                data_dict.update(
                    run_measurement(
                        b,
                        measurement_settings,
                        plot=plot_measurement,
                    )
                )
                data_dict.update(measurement_settings)

            data_dict.update(measurement_settings)

            if len(save_dict.items()) == 0:
                save_dict = data_dict
            else:
                save_dict = update_dict(save_dict, data_dict)

    return save_dict


def build_array(
    data_dict: dict, parameter_z: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x: np.ndarray = data_dict.get("x")[0][:, 0] * 1e6
    y: np.ndarray = data_dict.get("y")[0][:, 0] * 1e6
    z: np.ndarray = data_dict.get(parameter_z)

    zarray = z.reshape((len(y), len(x)), order="F")

    return x, y, zarray


def plot_slice(
    ax: Axes,
    data_dict: dict,
    parameter_z: str,
) -> Axes:
    sweep_parameter_x: str = data_dict.get("sweep_parameter_x")
    sweep_parameter_y: str = data_dict.get("sweep_parameter_y")
    x, y, zarray = build_array(data_dict, parameter_z)
    cmap = plt.cm.viridis(np.linspace(0, 1, 200))
    xfit, yfit = get_fitting_points(x, y, zarray)
    for i in range(len(x)):
        plot_parameter(ax, y, zarray[:, i], label=f"{sweep_parameter_x} = {x[i]:.1f}")
    # if len(x) == 1 and len(y) > 1:
    #     plot_parameter(ax, data_dict, B, C)
    # if len(y) == 1 and len(x) > 1:
    #     plot_parameter(ax, data_dict, A, C)
    ax.legend()
    ax.set_xlabel(sweep_parameter_y)
    ax.set_ylabel(parameter_z)
    return ax


if __name__ == "__main__":
    import scipy.io as sio

    data_dict = sio.loadmat(
        r"S:\SC\Measurements\SPG806\D6\A4\20241218_nMem_measure_enable_response\C2\SPG806_20241218_nMem_measure_enable_response_D6_A4_C2_2024-12-18 10-42-02.mat"
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_slice(ax, data_dict, "total_switches_norm")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_array(ax, data_dict, "total_switches_norm")
