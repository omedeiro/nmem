# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 14:01:31 2023

@author: omedeiro
"""

import time
from datetime import datetime
from time import sleep

import matplotlib as mpl
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
)
from nmem.measurement.cells import MITEQ_AMP_GAIN


def gauss(x: float, mu: float, sigma: float, A: float):
    return A * np.exp(-((x - mu) ** 2) / 2 / sigma**2)


def bimodal(
    x: float, mu1: float, sigma1: float, A1: float, mu2: float, sigma2: float, A2: float
):
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


def get_param_mean(param: np.ndarray) -> np.ndarray:
    if round(param[2], 5) > round(param[5], 5):
        prm = param[0:2]
    else:
        prm = param[3:5]
    return prm


def reject_outliers(data: np.ndarray, m: float = 2.0):
    ind = abs(data - np.mean(data)) < m * np.std(data)
    if len(ind[ind is False]) < 50:
        data = data[ind]
        print(f"Samples rejected {len(ind[ind is False])}")
        rejectInd = np.invert(ind)
    else:
        rejectInd = None
    return data, rejectInd


def update_dict(dict1: dict, dict2: dict):
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


def write_dict_to_file(file_path: str, save_dict: dict):
    with open(f"{file_path}_measurement_settings.txt", "w") as file:
        for key, value in save_dict.items():
            file.write(f"{key}: {value}\n")


def voltage2current(voltage: float, channel: int, measurement_settings: dict) -> float:
    cell = measurement_settings.get("cell")
    spice_device_current = measurement_settings.get("spice_device_current")
    spice_input_voltage = measurement_settings["HEATERS"][int(cell[1])].get(
        "spice_input_voltage"
    )
    spice_heater_voltage = measurement_settings["HEATERS"][int(cell[1])].get(
        "spice_heater_voltage"
    )
    heater_resistance = measurement_settings["CELLS"][cell].get("resistance_cryo")
    if channel == 1:
        current = spice_device_current * (voltage / spice_input_voltage)
    if channel == 2:
        current = (
            voltage * (spice_heater_voltage / spice_input_voltage) / heater_resistance
        )
    return current


def current2voltage(current: float, channel: int, measurement_settings: dict) -> float:
    cell = measurement_settings.get("cell")
    spice_device_current = measurement_settings.get("spice_device_current")
    spice_input_voltage = measurement_settings["HEATERS"][int(cell[1])].get(
        "spice_input_voltage"
    )
    spice_heater_voltage = measurement_settings["HEATERS"][int(cell[1])].get(
        "spice_heater_voltage"
    )
    heater_resistance = measurement_settings["HEATERS"][int(cell[1])].get(
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

    measurement_settings["channel_voltage"] = channel_voltage
    measurement_settings["channel_voltage_read"] = channel_voltage_read
    measurement_settings["enable_voltage"] = enable_voltage
    measurement_settings["wr_ratio"] = wr_ratio
    measurement_settings["ewr_ratio"] = ewr_ratio
    return measurement_settings


def calculate_threshold(read_zero_top, read_one_top):
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
    measurement_name = f"{date_str}_{measurement_name}"

    measurement_settings = {
        "measurement_name": measurement_name,
        "sample_name": sample_name,
        "cell": current_cell,
    }

    return measurement_settings, b


def setup_scope_bert(
    b: nTron,
    measurement_settings: dict,
    division_zero: float = 4.5,
    division_one: float = 9.5,
):
    scope_horizontal_scale = measurement_settings["scope_horizontal_scale"]
    scope_timespan = measurement_settings["scope_timespan"]
    scope_sample_rate = measurement_settings["scope_sample_rate"]
    threshold_read = measurement_settings.get("threshold_read", 100e-3)
    threshold_enab = measurement_settings.get("threshold_enab", 15e-3)
    num_meas = measurement_settings.get("num_meas")

    # b.inst.scope.set_deskew("F3", min(scope_timespan / 200, 5e-6))

    b.inst.scope.set_horizontal_scale(
        scope_horizontal_scale, -scope_horizontal_scale * 5
    )
    b.inst.scope.set_sample_rate(max(scope_sample_rate, 1e6))

    # b.inst.scope.set_measurement_clock_level("P1", "1", "Absolute", threshold_enab)
    # b.inst.scope.set_measurement_clock_level("P2", "1", "Absolute", threshold_enab)

    # b.inst.scope.set_measurement_clock_level("P1", "2", "Absolute", threshold_read)
    # b.inst.scope.set_measurement_clock_level("P2", "2", "Absolute", threshold_read)

    b.inst.scope.set_measurement_gate("P3", division_zero - 0.2, division_zero + 0.2)
    b.inst.scope.set_measurement_gate("P4", division_one - 0.2, division_one + 0.2)

    b.inst.scope.set_math_trend_values("F5", num_meas * 2)
    b.inst.scope.set_math_trend_values("F6", num_meas * 2)
    b.inst.scope.set_math_vertical_scale("F5", 100e-3, 300e-3)
    b.inst.scope.set_math_vertical_scale("F6", 100e-3, 300e-3)


def setup_scope_8bit_bert(
    b: nTron,
    measurement_settings: dict,
    division_zero: float = 4.5,
    division_one: float = 8.5,
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

    if enable_voltage > 200e-3:
        raise ValueError("enable voltage too high")

    if channel_voltage > 2.5:
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
    sleep(1)


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
    ww = measurement_settings["write_width"]
    rw = measurement_settings["read_width"]

    wr_ratio = (
        measurement_settings["write_current"] / measurement_settings["read_current"]
    )

    eww = measurement_settings["enable_write_width"]
    erw = measurement_settings["enable_read_width"]

    ew_phase = measurement_settings["enable_write_phase"]
    er_phase = measurement_settings["enable_read_phase"]

    ewr_ratio = (
        measurement_settings["enable_write_current"]
        / measurement_settings["enable_read_current"]
    )

    num_points = measurement_settings.get("num_points", 256)

    if wr_ratio >= 1:
        write_0 = create_waveforms_edge(width=ww, height=-1, edge=RISING_EDGE)
        write_1 = create_waveforms_edge(width=ww, height=1, edge=RISING_EDGE)
        read_wave = create_waveforms_edge(
            width=rw, height=1 / wr_ratio, edge=RISING_EDGE
        )
    else:
        write_0 = create_waveforms_edge(width=ww, height=-wr_ratio, edge=RISING_EDGE)
        write_1 = create_waveforms_edge(width=ww, height=wr_ratio, edge=RISING_EDGE)
        read_wave = create_waveforms_edge(width=rw, height=1, edge=RISING_EDGE)

    if ewr_ratio >= 1:
        enable_write = create_waveforms_edge(
            width=eww, height=1, phase=ew_phase, edge=0
        )
        enable_read = create_waveforms_edge(
            width=erw, height=1 / ewr_ratio, phase=er_phase, edge=0
        )
    else:
        enable_write = create_waveforms_edge(
            width=eww, height=ewr_ratio, phase=ew_phase, edge=0
        )
        enable_read = create_waveforms_edge(width=erw, height=1, phase=er_phase, edge=0)

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


def replace_bit(bitmsg, i, bit):
    return bitmsg[:i] + bit + bitmsg[i + 1 :]


def get_traces(b: nTron, scope_samples: int = 5000):
    # b.inst.scope.set_sample_mode('Realtime')
    sleep(1)
    b.inst.scope.set_trigger_mode("Single")
    sleep(0.1)
    trace_chan_in: np.ndarray = b.inst.scope.get_wf_data("C1")
    sleep(0.1)
    trace_chan_out: np.ndarray = b.inst.scope.get_wf_data("C2")
    sleep(0.1)
    trace_enab: np.ndarray = b.inst.scope.get_wf_data("C4")

    trace_chan_in = np.resize(trace_chan_in, (2, scope_samples))
    trace_chan_out = np.resize(trace_chan_out, (2, scope_samples))
    trace_enab = np.resize(trace_enab, (2, scope_samples))

    b.inst.scope.set_trigger_mode("Normal")
    sleep(1e-2)

    # try:
    #     time_est = round(b.inst.scope.get_parameter_value("P2"), 8)
    #     b.inst.scope.set_math_vertical_scale("F1", 50e-9, time_est)
    #     b.inst.scope.set_math_vertical_scale("F2", 50e-9, time_est)
    # except Exception:
    #     sleep(1e-4)

    return (
        trace_chan_in,
        trace_chan_out,
        trace_enab,
    )


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
    plt.sca(ax)
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(message):
        plt.text(
            i + 0.5,
            axheight * 0.85,
            message_dict[bit],
            ha="center",
            va="center",
            fontsize=14,
        )

    return ax


def plot_histogram(
    ax: Axes, y: np.ndarray, label: str, color: str, previous_params: list = [0]
) -> tuple:
    y = y[y > 300]

    if len(previous_params) == 1:
        ax, param, full_params = plot_hist_bimodal(ax, y, label=f"{label}", color=color)

    else:
        # expected = (800, 8, 0.05, 1000, 8, 0.05)
        ax, param, full_params = plot_hist_bimodal(
            ax, y, label=f"{label}", color=color, expected=previous_params
        )
    return ax, param, full_params


def plot_hist_bimodal(
    ax: Axes,
    y: np.ndarray,
    label: str,
    color: str,
    expected: tuple = (700, 10, 0.01, 1040, 10, 0.05),
):
    binwidth = 4
    h, edges, _ = ax.hist(
        y,
        bins=range(300, 1200 + binwidth, binwidth),
        label=label,
        color=color,
        density=True,
    )
    binwidth = np.diff(edges).mean()
    edges = (edges[1:] + edges[:-1]) / 2  # for len(x)==len(y)
    x = np.linspace(edges[1], edges[-1], 2000)

    try:
        full_params, cov = bimodal_fit(edges, h, expected)
        ax.plot(x, bimodal(x, *full_params), color="k", linewidth=1)
        lbl = "b"
        params = get_param_mean(full_params)

    except Exception:
        try:
            full_params, cov = bimodal_fit(edges, h, (700, 10, 0.02, 1040, 10, 0.02))
            lbl = "b"
            ax.plot(x, bimodal(x, *full_params), color="k", linewidth=1)
            lbl = "b"
            params = get_param_mean(full_params)

        except Exception:
            full_params, cov = bimodal_fit(edges, h, (700, 10, 0.02, 750, 10, 0.02))
            lbl = "b"
            params = get_param_mean(full_params)

    if (abs(np.array([full_params[2], full_params[5]])) < 1e-5).any() is True:
        try:
            ax.plot(x, bimodal(x, *params), color="k", linewidth=1)
            full_params = params
            params = get_param_mean(params)
            lbl = "b"
        except Exception:
            params = norm.fit(y)
            pdf = norm.pdf(x, *params) * h.sum() * binwidth
            ax.plot(x, pdf, color="k", linewidth=1)
            full_params = np.array(
                [params[0], params[1], expected[2], params[0], params[1], expected[5]]
            )
            lbl = "g"

    if (np.array([full_params[1], full_params[4]]) < 3).any() is True:
        try:
            params = norm.fit(y)
            expected = np.array([params[0] - 20, 12, 0.05, params[0] + 20, 10, 0.05])
            params, cov = bimodal_fit(edges, h, expected)
            ax.plot(x, bimodal(x, *params), color="k", linewidth=1)
            lbl = "b"
            full_params = np.array(
                [params[0], params[1], expected[2], params[0], params[1], expected[5]]
            )
        except Exception:
            params = norm.fit(y)
            pdf = norm.pdf(x, *params) * h.sum() * binwidth
            ax.plot(x, pdf, color="k", linewidth=1)
            full_params = np.array(
                [params[0], params[1], expected[2], params[0], params[1], expected[5]]
            )
            lbl = "g"

    ax.plot(
        [params[0] + abs(params[1]), params[0] + abs(params[1])],
        [0, 0.1],
        color="k",
        linewidth=0.5,
        label=f"{lbl}{abs(params[1]):.2f}",
    )

    ax.plot(
        [params[0] - abs(params[1]), params[0] - abs(params[1])],
        [0, 0.1],
        color="k",
        linewidth=0.5,
    )

    # ax.plot([full_params[0], full_params[0]], [0, 0.1], color='b', linewidth=0.5)
    # ax.plot([full_params[3], full_params[3]], [0, 0.1], color='b', linewidth=0.5)

    return ax, params, full_params


# def plot_hist(ax, y, label, color):
#     binwidth=3
#     h, edges, _ = ax.hist(y, bins=range(500, 1200 + binwidth, binwidth), color=color,  density = True)
#     param = norm.fit(y)   # Fit a normal distribution to the data
#     binwidth = np.diff(edges).mean()
#     edges = (edges[1:]+edges[:-1])/2 # for len(x)==len(y)
#     x = np.linspace(edges[1], edges[-1], 2000)
#     pdf = norm.pdf(x, *param)*h.sum()*binwidth
#     ax.plot(x, pdf, color = 'g', linewidth=1)
#     return ax, param


def plot_waveforms(
    data_dict: dict, measurement_settings: dict, previous_params: list = [0]
):
    i0 = data_dict["i0"]
    i1 = data_dict["i1"]
    bitmsg_channel = measurement_settings["bitmsg_channel"]
    bitmsg_enable = measurement_settings["bitmsg_enable"]

    trace_chan_in = data_dict["trace_chan_in"]
    trace_chan_out = data_dict["trace_chan_out"]
    trace_enab = data_dict["trace_enab"]

    numpoints = int((len(trace_chan_in[1]) - 1) / 2)

    ax1 = plt.subplot(411)
    # ax0.plot(
    #     trace_chan_in[0] * 1e6,
    #     trace_chan_in[1] * 1e3 / 50 - 6,
    #     label=f"peak current = {max(trace_chan_in[1][0:1000]*1e3/50):.1f}mA peak voltage = {max(trace_chan_in[1][0:1000]*1e3):.1f}mV",
    #     color="C7",
    # )
    # ax0.legend(loc=3)
    # plt.ylim((-12, 12))
    # plt.ylabel("splitter current (mA)")

    # ax1 = ax0.twinx()
    ax1.plot(
        trace_chan_out[0] * 1e6,
        trace_chan_out[1] * 1e3,
        label="channel",
        color="C4",
    )
    ax1.legend(loc=4)
    plt.ylim((-700, 700))
    plt.ylabel("volage (mV)")

    ax2 = plt.subplot(412)
    ax2.plot(trace_enab[0] * 1e6, trace_enab[1] * 1e3, label="enable", color="C1")
    ax2.legend(loc=1)
    plt.ylabel("volage (mV)")
    plt.xlabel("time (us)")
    plt.ylim((0, 300))
    plt.xlim((0, 10))

    ax3 = plt.subplot(413)
    y0 = i0[i0 > 0] * 1e6
    y0 = y0[y0 < 1100]

    y1 = i1[i1 > 0] * 1e6
    y1 = y1[y1 < 1100]

    if len(previous_params) > 1:
        pparams0 = previous_params[0]
        pparams1 = previous_params[1]

    else:
        pparams0 = [0]
        pparams1 = [0]

    ax3, param0, full_params0 = plot_histogram(
        ax3, y0, label="READ 0", color="C0", previous_params=pparams0
    )
    ax3, param1, full_params1 = plot_histogram(
        ax3, y1, label="READ 1", color="C2", previous_params=pparams1
    )
    plt.xlim([300, 1200])
    plt.ylim([-0.01, 0.08])

    distance = param0[0] - param1[0]
    # ydiff = pdf0-pdf1
    # np.mean([pdf0[ydiff==min(ydiff)],    pdf1[ydiff==min(ydiff)]])
    ber_est = 0

    plt.xlabel("read current (uA)")
    plt.ylabel("pdf")
    # plt.xlim([150, 350])
    ax3.legend(loc=2)

    ax4 = plt.subplot(414)
    ax4 = plot_trend(ax4, i0[i0 > 0] * 1e6, label="READ 0", color="C0", params=param0)
    ax4 = plot_trend(ax4, i1[i1 > 0] * 1e6, label="READ 1", color="C2", params=param1)
    plt.ylim([300, 1200])
    plt.ylabel("read current (uA)")
    plt.xlabel("sample")

    channel_voltage = measurement_settings["channel_voltage"]
    enable_voltage = measurement_settings["enable_voltage"]
    read_current = measurement_settings["read_current"]
    write_current = measurement_settings["write_current"]
    enable_read_current = measurement_settings["enable_read_current"]
    enable_write_current = measurement_settings["enable_write_current"]

    plt.suptitle(
        f"Write Current:{write_current*1e6:.1f}uA, Read Current:{read_current*1e6:.0f}uA, "
        "\n enable_write_current:{enable_write_current*1e6:.1f}uA, enable_read_current:{enable_read_current*1e6:.1f}uA"
        "\n Vcpp:{channel_voltage*1e3:.1f}mV, Vepp:{enable_voltage*1e3:.1f}mV, Channel_message: {bitmsg_channel}, Channel_enable: {bitmsg_enable}"
    )

    plt.tight_layout()

    saves = 0
    if saves == 1:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig(f"figure_{timestamp}.png")

    plt.show()

    return distance, full_params0, full_params1


def plot_trend(
    ax: Axes,
    y: np.ndarray,
    label: str,
    color: str,
    x: np.ndarray = None,
    params: tuple = None,
    m: float = 1,
):
    if x is None:
        x = np.arange(0, len(y), 1)

    ax.plot(x, y, ls="none", marker="o", label=label, color=color)
    if len(x) > 1:
        if params is not None:
            ax.plot(
                [x[0], x[-1]],
                np.tile(params[0] + params[1] * m, 2),
                color=np.subtract(1, mpl.colors.to_rgb(color)),
                linewidth=2,
                label=f"{label}",
            )
            ax.plot(
                [x[0], x[-1]],
                np.tile(params[0] - params[1] * m, 2),
                color=np.subtract(1, mpl.colors.to_rgb(color)),
                linewidth=2,
            )

    # if ind is not None:
    #     ax.plot(x[ind], y[ind], ls='none', marker='x', markersize=15, color=color)
    # print(ind)

    return ax


# def plot_waveforms_lite(data0, data1, data2, i0, i1, bitmsg_channel='1R0R', bitmsg_enable = 'WRWR'):
#     numpoints = int((len(data0[1])-1)/2)
#     ax3 = plt.axes()

#     ax3.hist(i0[i0>0]*1e6, 30, label='READ 0 ', color='C0')
#     ax3.hist(i1[i1>0]*1e6, 30, label='READ 1', color='C2')
#     plt.xlabel('read current (uA)')
#     plt.ylabel('count')
#     # plt.xlim([200, 1000])
#     ax3.legend(loc=2)
#     plt.suptitle(
#          f'Vpp:{channel_voltage*1e3:.1f}mV, Read Current:{read_current*1e6:.0f}uA, Write Current:{write_current*1e6:.1f}, enable_current:{enable_read_current*1e6:.1f} \n Channel_message: {bitmsg_channel}, Channel_enable: {bitmsg_enable}')

#     plt.show()


def plot_ber(x: np.ndarray, y: np.ndarray, ber: np.ndarray):
    dx = x[1] - x[0]
    xstart = x[0]
    xstop = x[-1]
    dy = y[1] - y[0]
    ystart = y[0]
    ystop = y[-1]

    cmap = plt.get_cmap("RdBu", 100).reversed()
    plt.imshow(
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
    # plt.xticks(np.arange(xstart*1e3,xstop*1e3, len(x)), rotation=45)
    # plt.yticks(np.arange(ystart*1e3,ystop*1e3, len(y)))
    plt.colorbar()


def plot_waveforms_bert(data_dict: dict, measurement_settings: dict):
    trace_chan_in = data_dict.get("trace_chan_in", [0, 0])
    trace_chan_out = data_dict.get("trace_chan_out", [0, 0])
    trace_enab = data_dict.get("trace_enab", [0, 0])
    read_zero_top = data_dict.get("read_zero_top", [0, 0])
    read_one_top = data_dict.get("read_one_top", [0, 0])

    channel_voltage = measurement_settings.get("channel_voltage", 0)
    enable_voltage = measurement_settings.get("enable_voltage", 0)
    read_current = measurement_settings.get("read_current", 0)
    write_current = measurement_settings.get("write_current", 0)
    enable_read_current = measurement_settings.get("enable_read_current", 0)
    enable_write_current = measurement_settings.get("enable_write_current", 0)
    bitmsg_channel = measurement_settings.get("bitmsg_channel", 0)
    bitmsg_enable = measurement_settings.get("bitmsg_enable", 0)
    measurement_name = measurement_settings.get("measurement_name", 0)
    sample_name = measurement_settings.get("sample_name", 0)
    scope_timespan = measurement_settings.get("scope_timespan", 0)
    threshold_bert = measurement_settings.get("threshold_enforced")

    numpoints = int((len(trace_chan_in[1]) - 1) / 2)
    cmap = plt.cm.viridis(np.linspace(0, 1, 200))
    C1 = 45
    C2 = 135
    ax0 = plt.subplot(411)
    ax0.plot(
        trace_chan_in[0] * 1e6,
        trace_chan_in[1] * 1e3,
        label="input signal",
        color=cmap[C1, :],
    )
    ax0.legend(loc=4)
    plt.ylabel("voltage (mV)")
    plt.xlabel("time (us)")
    plt.xlim((0, scope_timespan * 1e6))
    plt.ylim((-200, 1200))
    ax0 = plot_message(ax0, bitmsg_channel)

    ax1 = plt.subplot(413)
    ax1.plot(
        trace_chan_out[0] * 1e6,
        trace_chan_out[1] * 1e3,
        label="channel",
        color=cmap[C1, :],
    )
    ax1.hlines(
        threshold_bert * 1e3, 0, len(trace_chan_in[0]), color="r", label="threshold"
    )
    plt.xlabel("time (us)")
    plt.ylabel("volage (mV)")
    ax1.legend(loc=4)
    plt.xlim((0, scope_timespan * 1e6))
    plt.ylim((-200, 700))
    ax1 = plot_message(ax1, bitmsg_channel)

    ax2 = plt.subplot(412)
    ax2.plot(
        trace_enab[0] * 1e6, trace_enab[1] * 1e3, label="enable", color=cmap[C1, :]
    )
    ax2.legend(loc=4)
    plt.xlabel("time (us)")
    plt.ylabel("volage (mV)")
    plt.xlim((0, scope_timespan * 1e6))
    plt.ylim((-50, 200))
    ax2 = plot_message(ax2, bitmsg_enable)

    ax3 = plt.subplot(414)
    ax3 = plot_trend(ax3, read_zero_top, label="READ 0", color=cmap[C1, :])
    ax3 = plot_trend(ax3, read_one_top, label="READ 1", color=cmap[C2, :])
    ax3.hlines(threshold_bert, 0, len(read_zero_top), color="r", label="threshold")
    plt.ylim([0, 0.6])
    plt.ylabel("read voltage (V)")
    plt.xlabel("sample")
    ax3.legend(loc=4)

    plt.suptitle(
        f"{sample_name} -- {measurement_name} \n Vcpp:{channel_voltage*1e3:.1f}mV, Vepp:{enable_voltage*1e3:.1f}mV, Write Current:{write_current*1e6:.1f}, Read Current:{read_current*1e6:.0f}uA, \n enable_write_current:{enable_write_current*1e6:.1f}, enable_read_current:{enable_read_current*1e6:.1f} \n Channel_message: {bitmsg_channel}, Channel_enable: {bitmsg_enable}"
    )

    plt.tight_layout()

    plt.show()


def plot_parameter(
    data_dict: dict,
    x_name: str,
    y_name: str,
    xindex: int = 0,
    yindex: int = 0,
    ax: Axes = None,
    **kwargs,
):
    x_length = data_dict["x"].shape[1]
    y_length = data_dict["y"].shape[1]

    if not ax:
        fig, ax = plt.subplots()

    x = data_dict[x_name][xindex].flatten().reshape(x_length, y_length).squeeze()
    y = data_dict[y_name][yindex].flatten().reshape(x_length, y_length)
    ax.plot(x.flatten(), y.flatten(), marker="o", label=y_name, **kwargs)
    plt.xticks(rotation=60)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    measurement_name = data_dict["measurement_name"]
    sample_name = data_dict["sample_name"]

    time_str = data_dict["time_str"]

    channel_voltage = data_dict["channel_voltage"].flatten()
    enable_voltage = data_dict["enable_voltage"].flatten()
    read_current = data_dict["read_current"].flatten()
    write_current = data_dict["write_current"].flatten()
    enable_read_current = data_dict["enable_read_current"].flatten()
    enable_write_current = data_dict["enable_write_current"].flatten()
    bitmsg_channel = data_dict["bitmsg_channel"]
    bitmsg_enable = data_dict["bitmsg_enable"]

    plt.suptitle(
        f"{sample_name} -- {measurement_name} \n {time_str} \n Vcpp:{channel_voltage[0]*1e3:.1f}mV, Vepp:{enable_voltage[0]*1e3:.1f}mV, Write Current:{write_current[0]*1e6:.1f}uA, Read Current:{read_current[0]*1e6:.0f}uA, \n enable_write_current:{enable_write_current[0]*1e6:.1f}, enable_read_current:{enable_read_current[0]*1e6:.1f} \n Channel_message: {bitmsg_channel}, Channel_enable: {bitmsg_enable}"
    )

    # plt.suptitle(
    #     f'{sample_name[0]} -- {measurement_name[0]} \n {time_str}')

    print(f"min {y_name} at {x_name} = {x[y.argmin()]:3.2e}")
    return ax


def plot_array(
    data_dict: dict,
    c_name: str,
    x_name: str = "x",
    y_name: str = "y",
    ax: Axes = None,
    cmap=None,
    norm=True,
):
    if not ax:
        fig, ax = plt.subplots()

    x = data_dict["x"][0][:, 0] * 1e6
    y = data_dict["y"][0][:, 0] * 1e6
    c = data_dict[c_name]

    ctotal = c.reshape((len(y), len(x)), order="F")

    dx = x[1] - x[0]
    xstart = x[0]
    xstop = x[-1]
    dy = y[1] - y[0]
    ystart = y[0]
    ystop = y[-1]

    if not cmap:
        cmap = plt.get_cmap("RdBu", 100).reversed()

    plt.imshow(
        ctotal,
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

    # print(f'{dx}, {xstart}, {xstop}, {dy}, {ystart}, {ystop}')
    plt.xticks(np.linspace(xstart, xstop, len(x)), rotation=45)
    plt.yticks(np.linspace(ystart, ystop, len(y)))
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    xv, yv = np.meshgrid(x, y, indexing="ij")

    if norm:
        plt.clim([0, 1])
        cbar = plt.colorbar(ticks=np.linspace(0, 1, 11))
    else:
        plt.clim([0, c.max()])
        cbar = plt.colorbar()
    cbar.ax.set_ylabel(c_name, rotation=270)
    plt.contour(xv, yv, np.reshape(ctotal, (len(y), len(x))).T, [0.05, 0.1])
    measurement_name = data_dict["measurement_name"]
    sample_name = data_dict["sample_name"]
    time_str = data_dict["time_str"]

    plt.suptitle(f"{sample_name} -- {measurement_name} \n {time_str}")

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


def run_realtime_bert(b: nTron, measurement_settings: dict, channel="F5") -> dict:
    num_meas = measurement_settings.get("num_meas")

    b.inst.scope.set_trigger_mode("Normal")
    sleep(0.5)
    b.inst.scope.clear_sweeps()
    sleep(0.1)
    with tqdm(total=num_meas) as pbar:
        while b.inst.scope.get_num_sweeps(channel) < num_meas:
            sleep(0.1)
            n = b.inst.scope.get_num_sweeps(channel)
            pbar.update(n - pbar.n)
            # print(f"sampling...{n} / {num_meas}")

    b.inst.scope.set_trigger_mode("Stop")

    # This assumes that the measurements are always zero then one.
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

    # Find the difference between the highest and lowest values in the read top arrays
    difference = read_one_top - read_zero_top
    max_diff = difference.max()
    if max_diff > 0.3:
        print(f"Max difference: {max_diff*1e3:.2f} mV")
        threshold = calculate_threshold(read_zero_top, read_one_top)
        print(f"Calculated Threshold: {threshold*1e3:.2f} mV")
    else:
        threshold = measurement_settings.get(
            "threshold_bert", measurement_settings["threshold_enforced"]
        )
        print(f"Max difference: {max_diff*1e3:.2f} mV")
        print(f"Default Threshold: {threshold*1e3:.2f} mV")

    measurement_settings["threshold_enforced"] = threshold
    print(f"Enforced Threshold: {threshold*1e3:.2f} mV")

    # READ 1: above threshold (voltage)
    write_0_read_1 = np.array([(read_zero_top > threshold).sum()])

    # READ 0: below threshold (no voltage)
    write_1_read_0 = np.array([(read_one_top < threshold).sum()])

    write_0_read_1_norm = write_0_read_1 / (num_meas * 2)
    write_1_read_0_norm = write_1_read_0 / (num_meas * 2)

    result_dict = {
        "write_0_read_1": write_0_read_1,
        "write_1_read_0": write_1_read_0,
        "write_0_read_1_norm": write_0_read_1_norm,
        "write_1_read_0_norm": write_1_read_0_norm,
        "read_zero_top": read_zero_top,
        "read_one_top": read_one_top,
    }
    return result_dict, measurement_settings


def run_sequence_bert(b: nTron, measurement_settings: dict):
    if measurement_settings["num_meas"]:
        num_meas = measurement_settings["num_meas"]
    else:
        num_meas = 100

    b.inst.scope.clear_sweeps()
    sleep(0.1)
    b.inst.scope.set_trigger_mode("Single")
    sleep(1)
    while b.inst.scope.get_trigger_mode() != "Stopped\n":
        sleep(1)
        print("sampling...")

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

    threshold = measurement_settings.get("threshold_bert", 150e-3)

    # READ 1: above threshold (voltage)
    write_0_read_1 = (read_zero_top > threshold).sum()

    # READ 0: below threshold (no voltage)
    write_1_read_0 = (read_one_top < threshold).sum()

    write_0_read_1_norm = write_0_read_1 / (num_meas * 2)
    write_1_read_0_norm = write_1_read_0 / (num_meas * 2)

    print(f"w0r1: {write_0_read_1}, w1r0: {write_1_read_0}")
    return (
        read_zero_top,
        read_one_top,
        write_0_read_1,
        write_1_read_0,
        write_0_read_1_norm,
        write_1_read_0_norm,
    )


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
    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

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
    save_traces: bool = True,
    plot: bool = False,
    ramp_read: bool = False,
):
    measurement_settings = calculate_voltage(measurement_settings)
    bitmsg_channel: str = measurement_settings.get("bitmsg_channel")
    bitmsg_enable: str = measurement_settings.get("bitmsg_enable")
    sample_rate: float = measurement_settings.get("sample_rate")
    channel_voltage: float = measurement_settings.get("channel_voltage")
    enable_voltage: float = measurement_settings.get("enable_voltage")
    scope_samples: int = int(measurement_settings.get("scope_num_samples"))
    num_meas: int = measurement_settings.get("num_meas")

    ######################################################

    setup_waveform(b, measurement_settings)

    b.inst.awg.set_output(True, 1)
    b.inst.awg.set_output(True, 2)

    b.inst.scope.clear_sweeps()

    meas_dict, measurement_settings = run_realtime_bert(b, measurement_settings)

    trace_chan_in, trace_chan_out, trace_enab = get_traces(b, scope_samples)

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    bit_error_rate = calculate_bit_error_rate(
        meas_dict["write_1_read_0"], meas_dict["write_0_read_1"], num_meas
    )
    DATA_DICT = {
        **meas_dict,
        "trace_chan_in": trace_chan_in,
        "trace_chan_out": trace_chan_out,
        "trace_enab": trace_enab,
        "bit_error_rate": bit_error_rate,
    }
    print(f"Bit Error Rate: {bit_error_rate:.2e}")
    if plot:
        plot_waveforms_bert(DATA_DICT, measurement_settings)

    return DATA_DICT, measurement_settings


def run_sweep(
    b: nTron,
    measurement_settings: dict,
    parameter_x: str,
    parameter_y: str,
    save_traces: bool = False,
    plot_measurement=False,
):
    save_dict = {}

    setup_scope_bert(b, measurement_settings)

    for x in measurement_settings["x"]:
        for y in measurement_settings["y"]:
            measurement_settings[parameter_x] = x
            measurement_settings[parameter_y] = y
            cell = b.properties["Save File"]["cell"]
            slope = measurement_settings["CELLS"][cell]["slope"]
            intercept = measurement_settings["CELLS"][cell]["intercept"]
            cell_dict = measurement_settings["CELLS"][cell]
            write_critical_current = calculate_critical_current(
                measurement_settings["enable_write_current"] * 1e6, cell_dict
            )
            read_critical_current = calculate_critical_current(
                measurement_settings["enable_read_current"] * 1e6, cell_dict
            )
            max_heater_current = -intercept / slope
            # print(f"Write Current: {measurement_settings['write_current']:.2f}")
            # print(f"Write Critical Current: {write_critical_current:.2f}")
            # print(f"Read Current: {measurement_settings['read_current']:.2f}")
            # print(f"Read Critical Current: {read_critical_current:.2f}")

            data_dict, measurment_settings = run_measurement(
                b,
                measurement_settings,
                save_traces,
                plot=plot_measurement,
                ramp_read=False,
            )

            data_dict.update(measurement_settings)
            enable_write_power = calculate_heater_power(
                measurement_settings["enable_write_current"],
                measurement_settings["CELLS"][cell]["resistance_cryo"],
            )
            enable_read_power = calculate_heater_power(
                measurement_settings["enable_read_current"],
                measurement_settings["CELLS"][cell]["resistance_cryo"],
            )
            param_dict = {
                "Write Current [uA]": [measurement_settings["write_current"] * 1e6],
                "Read Current [uA]": [measurement_settings["read_current"] * 1e6],
                "Enable Write Current [uA]": [
                    measurement_settings["enable_write_current"] * 1e6
                ],
                "Enable Read Current [uA]": [
                    measurement_settings["enable_read_current"] * 1e6
                ],
                "Write Critical Current [uA]": [write_critical_current],
                "Write Enable Fraction": [
                    measurement_settings["enable_write_current"]
                    * 1e6
                    / max_heater_current
                ],
                "Read Critical Current [uA]": [read_critical_current],
                "Read Enable Fraction": [
                    measurement_settings["enable_read_current"]
                    * 1e6
                    / max_heater_current
                ],
                "Max Heater Current [uA]": [max_heater_current],
                "Write / Read Ratio": [data_dict["wr_ratio"]],
                "Enable Write / Read Ratio": [data_dict["ewr_ratio"]],
                "Enable Write Power [uW]": [enable_write_power * 1e6],
                "Enable Read Power [uW]": [enable_read_power * 1e6],
                "Write 0 Read 1": [data_dict["write_0_read_1"]],
                "Write 1 Read 0": [data_dict["write_1_read_0"]],
                "Bit Error Rate": [data_dict["bit_error_rate"]],
            }
            if measurement_settings["write_current"] > 0:
                param_dict.update(
                    {
                        "Write Bias Fraction": [
                            measurement_settings["write_current"]
                            / (write_critical_current * 1e-6)
                        ],
                    }
                )

            if measurement_settings["read_current"] > 0:
                param_dict.update(
                    {
                        "Read Bias Fraction": [
                            measurement_settings["read_current"]
                            / (read_critical_current * 1e-6)
                        ],
                    }
                )
            pd.set_option("display.float_format", "{:.3f}".format)
            param_df = pd.DataFrame(param_dict.values(), index=param_dict.keys())
            param_df.columns = ["Value"]

            enable_write_power = calculate_heater_power(
                measurement_settings["enable_write_current"],
                measurement_settings["CELLS"][cell]["resistance_cryo"],
            )
            enable_read_power = calculate_heater_power(
                measurement_settings["enable_read_current"],
                measurement_settings["CELLS"][cell]["resistance_cryo"],
            )
            print(f"Enable Write Power [uW]: {enable_write_power*1e6:.2f}")
            print(f"Enable Read Power [uW]: {enable_read_power*1e6:.2f}")

            if data_dict["bit_error_rate"] < 0.01:
                write_voltage_high = np.mean(data_dict["read_one_top"][1])
                write_voltage_low = np.mean(data_dict["read_zero_top"][1])
                print(f"Write Voltage High: {write_voltage_high:.2f}")
                print(f"Write Voltage Low: {write_voltage_low:.2f}")

                switch_voltage = write_voltage_high - write_voltage_low
                switch_voltage_preamp = switch_voltage / 10 ** (MITEQ_AMP_GAIN / 20)
                print(f"Switch Voltage Amp [mV]: {switch_voltage*1e3:.2f}")
                print(f"Switch Voltage Preamp [mV]: {switch_voltage_preamp*1e3:.4f}")

                switch_power = (
                    switch_voltage_preamp * measurement_settings["read_current"]
                )
                print(f"Read Power [nW]: {switch_power*1e9:.2f}")
                switch_impedance = (
                    switch_voltage_preamp / measurement_settings["read_current"]
                )
                write_power = (
                    switch_impedance * measurement_settings["write_current"] ** 2
                )
                print(f"Write Power [nW]: {write_power*1e9:.2f}")
                print(f"Switch Impedance [Ohm]: {switch_impedance:.2f}")

            if len(save_dict.items()) == 0:
                save_dict = data_dict
            else:
                save_dict = update_dict(save_dict, data_dict)

    final_dict = {
        **measurement_settings,
        **save_dict,
    }
    return save_dict


def run_sweep_subset(
    b: nTron,
    measurement_settings: dict,
    parameter_x: str,
    parameter_y: str,
    save_traces: bool = False,
    plot_measurement=False,
):
    save_dict = {}
    for idx, x in enumerate(measurement_settings["x"]):
        for y in measurement_settings["y"]:
            measurement_settings[parameter_x] = x
            measurement_settings[parameter_y] = y
            measurement_settings = calculate_voltage(measurement_settings)

            if (
                y > measurement_settings["y_subset"][idx][0]
                and y < measurement_settings["y_subset"][idx][-1]
            ):
                data_dict = run_measurement(
                    b,
                    measurement_settings,
                    save_traces,
                    plot=plot_measurement,
                    ramp_read=False,
                )
                data_dict.update(measurement_settings)
                print(f"threshold_bert: {measurement_settings['threshold_bert']:.2e}")
                print(f"bit_error_rate: {data_dict['bit_error_rate']}")
                print(f"write_0_read_1: {data_dict['write_0_read_1']}")
                print(f"write_1_read_0: {data_dict['write_1_read_0']}")
                print("-")
                print(f"write_current [uA]: {data_dict['write_current']*1e6:.2f}")
                print(f"read_current [uA]: {data_dict['read_current']*1e6:.2f}")
                print(
                    f"enable_write_current [uA]: {data_dict['enable_write_current']*1e6:.2f}"
                )
                print(
                    f"enable_read_current [uA]: {data_dict['enable_read_current']*1e6:.2f}"
                )
                print("-")
                print(f"wr_ratio: {data_dict['wr_ratio']}")
                print(f"ewr_ratio: {data_dict['ewr_ratio']}")
                print("-----------------")
            else:
                data_dict = {
                    "trace_chan_in": np.empty(
                        (2, measurement_settings["scope_num_samples"])
                    ),
                    "trace_chan_out": np.empty(
                        (2, measurement_settings["scope_num_samples"])
                    ),
                    "trace_enab": np.empty(
                        (2, measurement_settings["scope_num_samples"])
                    ),
                    "write_0_read_1": np.nan,
                    "write_1_read_0": np.nan,
                    "write_0_read_1_norm": np.nan,
                    "write_1_read_0_norm": np.nan,
                    "read_zero_top": np.empty((1, measurement_settings["num_meas"])),
                    "read_one_top": np.empty((1, measurement_settings["num_meas"])),
                    "channel_voltage": np.nan,
                    "enable_voltage": np.nan,
                    "bit_error_rate": np.nan,
                }

            data_dict.update(measurement_settings)

            if len(save_dict.items()) == 0:
                save_dict = data_dict
            else:
                save_dict = update_dict(save_dict, data_dict)

    final_dict = {
        **measurement_settings,
        **save_dict,
    }
    return save_dict


def plot_ber_sweep(
    save_dict: dict, measurement_settings: dict, file_path: str, A: str, B: str, C: str
) -> None:
    x = measurement_settings["x"]
    y = measurement_settings["y"]
    if len(x) > 1 and len(y) > 1:
        ax = plot_array(save_dict, C, A, B)
        plt.savefig(file_path + "_0.png")

        plot_array(
            save_dict,
            "write_0_read_1",
            A,
            B,
            cmap=plt.get_cmap("Reds", 100),
            norm=False,
        )
        plt.savefig(file_path + "_1.png")

        plt.show()
        plot_array(
            save_dict,
            "write_1_read_0",
            A,
            B,
            cmap=plt.get_cmap("Blues", 100),
            norm=False,
        )
        plt.savefig(file_path + "_2.png")

        plt.show()

    if len(x) == 1 and len(y) > 1:
        ax = plot_parameter(save_dict, B, C, color="#808080")
        ax = plot_parameter(
            save_dict, B, "write_0_read_1_norm", ax=ax, color=(0.68, 0.12, 0.1)
        )
        ax = plot_parameter(
            save_dict, B, "write_1_read_0_norm", ax=ax, color=(0.16, 0.21, 0.47)
        )
        plt.savefig(file_path + "_3.png")

    if len(y) == 1 and len(x) > 1:
        ax = plot_parameter(save_dict, A, C, color="#808080")
        ax = plot_parameter(
            save_dict, A, "write_0_read_1_norm", ax=ax, color=(0.68, 0.12, 0.1)
        )
        ax = plot_parameter(
            save_dict, A, "write_1_read_0_norm", ax=ax, color=(0.16, 0.21, 0.47)
        )
        plt.savefig(file_path + "_4.png")
