# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:17:34 2023

@author: omedeiro
"""

import time

import numpy as np
import qnnpy.functions.functions as qf
import qnnpy.functions.ntron as nt
from matplotlib import pyplot as plt

import nmem.measurement.functions as nm
from nmem.calculations.calculations import calculate_critical_current
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
    b = nt.nTron(CONFIG)

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
    measurement_name = f"{date_str}_nMem_ICE_ber"

    current_cell = b.properties["Save File"]["cell"]
    sample_name = [
        b.sample_name,
        b.device_type,
        b.device_name,
        current_cell,
    ]
    sample_name = str("-".join(sample_name))
    date_str = time.strftime("%Y%m%d")
    measurement_name = f"{date_str}_nMem_ICE_ber"

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 90,
        "read_width": 82,  #
        "enable_write_width": 36,
        "enable_read_width": 33,
        "enable_write_phase": -12,
        "enable_read_phase": -35,
        "bitmsg_channel": "N0RNR1RNRN",
        "bitmsg_enable": "NWNWEWNWEW",
    }

    current_settings = {
        "write_current": 45.213e-6,
        "read_current": 675.9e-6,
        "enable_write_current": 224.494e-6,
        "enable_read_current": 179.9e-6,
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }
    NUM_MEAS = 5000

    measurement_settings = {
        **waveform_settings,
        **current_settings,
        **scope_settings,
        "measurement_name": measurement_name,
        "sample_name": sample_name,
        "cell": current_cell,
        "CELLS": CELLS,
        "HEATERS": HEATERS,
        "num_meas": NUM_MEAS,
        "spice_device_current": SPICE_DEVICE_CURRENT,
        "x": 0,
        "y": 0,
        "threshold_bert": 0.2,
    }

    parameter_x = "enable_write_current"
    measurement_settings["x"] = np.array([224.494e-6])
    # measurement_settings["x"] = np.linspace(260e-6, 350e-6, 11)
    measurement_settings[parameter_x] = measurement_settings["x"][0]
    # measurement_settings["x"] = np.linspace(260e-6, 290e-6, 3)

    read_sweep = False
    if read_sweep:
        parameter_y = "read_current"
        measurement_settings["y"] = np.array([688e-6])
        # measurement_settings["y"] = np.linspace(610e-6, 650e-6, 11)
        # read_critical_current = calculate_critical_current(
        #     measurement_settings["enable_read_current"], CELLS[current_cell]
        # )
        # measurement_settings["y"] = np.linspace(
        #     read_critical_current * 0.75, read_critical_current * 1.0, 11
        # )
    else:
        parameter_y = "write_current"
        measurement_settings["y"] = np.array([45.213e-6])
        # measurement_settings["y"] = np.linspace(40e-6, 90e-6, 5)
        # write_critical_current = calculate_critical_current(
        #     measurement_settings["enable_write_current"], CELLS[current_cell]
        # )
        # measurement_settings["y"] = np.linspace(
        #     write_critical_current * 0.05, write_critical_current * 0.6, 15
        # )

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
