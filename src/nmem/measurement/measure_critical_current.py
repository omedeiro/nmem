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
from nmem.measurement.cells import (
    CELLS,
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

CONFIG = r"SPG806_config_ICE.yml"


if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_parameter_sweep"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 90,
        "read_width": 82,  #
        "enable_write_width": 36,
        "enable_read_width": 33,
        "enable_write_phase": -12,
        "enable_read_phase": -35,
        "bitmsg_channel": "N0NNRNNNRN",
        "bitmsg_enable": "NNNNENNNEW",
    }

    current_settings = {
        "write_current": 50e-6,
        "read_current": 600e-6,
        "enable_write_current": 10e-6,
        "enable_read_current": 10e-6,
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }
    NUM_MEAS = 100

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
            "threshold_bert": 0.2,
        }
    )
    parameter_x = "enable_read_current"
    measurement_settings["x"] = np.array([0])

    parameter_y = "read_current"
    measurement_settings["y"] = np.linspace(750e-6, 770e-6, 15)

    save_dict = nm.run_sweep(
        b, measurement_settings, parameter_x, parameter_y, plot_measurement=True
    )
    b.properties["measurement_settings"] = measurement_settings

    file_path, time_str = qf.save(b.properties, measurement_name, save_dict)
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
