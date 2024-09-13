# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:17:34 2023

@author: omedeiro
"""

import time

import numpy as np
import qnnpy.functions.functions as qf
from matplotlib import pyplot as plt

import nmem.measurement.functions as nm
from nmem.calculations.calculations import calculate_critical_current
from nmem.measurement.cells import (
    CELLS,
    CONFIG,
    FREQ_IDX,
    HEATERS,
    HORIZONTAL_SCALE,
    NUM_DIVISIONS,
    NUM_POINTS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    SPICE_DEVICE_CURRENT,
)

plt.rcParams["figure.figsize"] = [10, 12]


def read_sweep_scaled(measurement_settings, current_cell, num_points=15):
    read_critical_current = (
        calculate_critical_current(
            measurement_settings["enable_read_current"] * 1e6, CELLS[current_cell]
        )
        * 1e-6
    )
    measurement_settings["y"] = np.linspace(
        read_critical_current * 0.8, read_critical_current * 0.95, num_points
    )
    return measurement_settings


def write_sweep_scaled(measurement_settings, current_cell, num_points=15):
    write_critical_current = (
        calculate_critical_current(
            measurement_settings["enable_write_current"] * 1e6, CELLS[current_cell]
        )
        * 1e-6
    )
    measurement_settings["y"] = np.linspace(
        write_critical_current * 0.0, write_critical_current * 0.15, num_points
    )
    return measurement_settings


if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_parameter_sweep"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    current_cell = measurement_settings["cell"]

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 30,
        "read_width": 30,  #
        "enable_write_width": 30,
        "enable_read_width": 35,
        "enable_write_phase": 0,
        "enable_read_phase": -20,
        "bitmsg_channel": "N0NNRN1NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }

    current_settings = {
        "write_current": 60.160e-6,
        "read_current": 640.192e-6,
        "enable_write_current": 289.500e-6,
        "enable_read_current": 212.843e-6,
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }
    NUM_MEAS = 200
    NUM_POINTS = 21

    measurement_settings.update(
        {
            **waveform_settings,
            **current_settings,
            **scope_settings,
            "CELLS": CELLS,
            "HEATERS": HEATERS,
            "num_meas": NUM_MEAS,
            "spice_device_current": SPICE_DEVICE_CURRENT,
            "x": 0,
            "y": 0,
        }
    )
    parameter_x = "enable_write_current"
    measurement_settings["x"] = np.array([289.5e-6])
    measurement_settings[parameter_x] = measurement_settings["x"][0]

    read_sweep = True
    if read_sweep:
        parameter_y = "read_current"
        measurement_settings = read_sweep_scaled(
            measurement_settings, current_cell, NUM_POINTS
        )
    else:
        parameter_y = "write_current"
        measurement_settings = write_sweep_scaled(
            measurement_settings, current_cell, NUM_POINTS
        )
        # measurement_settings["y"] = np.array([29.108e-6])

    save_dict = nm.run_sweep(
        b, measurement_settings, parameter_x, parameter_y, plot_measurement=True
    )
    b.properties["measurement_settings"] = measurement_settings

    file_path, time_str = qf.save(
        b.properties, measurement_settings["measurement_name"], save_dict
    )
    save_dict["time_str"] = time_str
    nm.plot_ber_sweep(
        save_dict,
        measurement_settings,
        file_path,
        parameter_x,
        parameter_y,
        "bit_error_rate",
    )

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    nm.write_dict_to_file(file_path, measurement_settings)

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
