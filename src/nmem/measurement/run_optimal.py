# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:17:34 2023

@author: omedeiro
"""

import time

import qnnpy.functions.functions as qf
from matplotlib import pyplot as plt

import nmem.measurement.functions as nm
from nmem.measurement.cells import (
    CELLS,
    CONFIG,
    DEFAULT_SCOPE,
    FREQ_IDX,
    HEATERS,
    NUM_POINTS,
    SAMPLE_RATE,
    SPICE_DEVICE_CURRENT,
)

plt.rcParams["figure.figsize"] = [10, 12]

if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_parameter_sweep"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    current_cell = measurement_settings["cell"]

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 0,
        "read_width": 10,
        "enable_write_width": 4,
        "enable_read_width": 4,
        "enable_write_phase": -7,
        "enable_read_phase": -7,
        "bitmsg_channel": "N0NNRN1NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }

    current_settings = CELLS[current_cell]

    NUM_MEAS = 5000

    measurement_settings.update(
        {
            **waveform_settings,
            **current_settings,
            **DEFAULT_SCOPE,
            "CELLS": CELLS,
            "HEATERS": HEATERS,
            "num_meas": NUM_MEAS,
            "spice_device_current": SPICE_DEVICE_CURRENT,
            "sweep_parameter_x": "enable_read_current",
            "sweep_parameter_y": "read_current",
        }
    )
    nm.setup_scope_bert(
        b,
        measurement_settings,
        division_zero=(5.9, 6.5),
        division_one=(5.9, 6.5),
    )
    data_dict = nm.run_measurement(
        b,
        measurement_settings,
        plot=True,
    )

    file_path, time_str = qf.save(
        b.properties, data_dict.get("measurement_name"), data_dict
    )
    print(f"Bit Error Rate: {data_dict['bit_error_rate']}")

    nm.write_dict_to_file(file_path, measurement_settings)

    nm.set_awg_off(b)

    t2 = time.time()
    print(f"run time {(t2 - t1) / 60:.2f} minutes")
