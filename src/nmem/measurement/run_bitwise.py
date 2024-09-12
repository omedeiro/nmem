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
        "num_points": 64,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 60,
        "read_width": 60,  #
        "enable_write_width": 60,
        "enable_read_width": 60,
        "enable_write_phase": 0,
        "enable_read_phase": 0,
        "bitmsg_channel": "0R1R0R1R0R1R0R1R",
        "bitmsg_enable": "WEWEWEWEWNNNNNNE",
    }


    current_settings = CELLS[current_cell]

    NUM_MEAS = 500
    scope_settings = {
    "scope_horizontal_scale": 1e-6,
    "scope_timespan": 1e-6*10,
    "scope_num_samples": 1000,
    "scope_sample_rate": 1e3 / (1e-6*10),
}
    
    measurement_settings.update(
        {
            **waveform_settings,
            **current_settings,
            **DEFAULT_SCOPE,
            "CELLS": CELLS,
            "HEATERS": HEATERS,
            "num_meas": NUM_MEAS,
            "spice_device_current": SPICE_DEVICE_CURRENT,
            "x": 0,
            "y": 0,
        }
    )

    
    nm.setup_scope_8bit_bert(b, measurement_settings)

    nm.run_bitwise_measurement(b, measurement_settings)

