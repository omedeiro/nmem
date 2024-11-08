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
    HEATERS,
    HORIZONTAL_SCALE,
    NUM_DIVISIONS,
    NUM_POINTS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    SPICE_DEVICE_CURRENT,
)

plt.rcParams["figure.figsize"] = [10, 12]

if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_parameter_sweep"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    current_cell = measurement_settings["cell"]
    FREQ_IDX = 2

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 40,
        "read_width": 20,
        "enable_write_width": 40,
        "enable_read_width": 120,
        "enable_write_phase": 0,
        "enable_read_phase": 0,
        "bitmsg_channel": "NN0R1R0R1R0R1R0R1RNN",
        "bitmsg_enable": "NNWEWEWEWEWEWEWEWENN",
        "threshold_bert": 0.33,
        "threshold_enforced": 0.33,
    }

    current_settings = {
        "write_current": 30e-6,
        "read_current": 680e-6,
        "enable_write_current": 320e-6,
        "enable_read_current": 220e-6,
    }
    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }

    NUM_MEAS = 1000
    sweep_length = 11

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
    measurement_settings["x"] = np.array([measurement_settings[parameter_x]])
    # measurement_settings["x"] = np.linspace(250e-6, 340e-6, sweep_length)
    measurement_settings[parameter_x] = measurement_settings["x"][0]

    read_sweep = True
    if read_sweep:
        parameter_y = "read_current"
        measurement_settings = read_sweep_scaled(
            measurement_settings, current_cell, sweep_length, start=0.8, end=1.05
        )
        # measurement_settings["y"] = np.array([current_settings["read_current"]])
    else:
        parameter_y = "write_current"
        # measurement_settings = write_sweep_scaled(
        #     measurement_settings, current_cell, sweep_length
        # )
        measurement_settings["y"] = np.array([current_settings["write_current"]])

    save_dict = nm.run_sweep(
        b,
        measurement_settings,
        parameter_x,
        parameter_y,
        plot_measurement=True,
        division_zero=3.5,
        division_one=8.5,
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
