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
    CONFIG,
    DEFAULT_SCOPE,
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

if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_parameter_sweep"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    current_cell = measurement_settings["cell"]

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 90,
        "read_width": 82,  #
        "enable_write_width": 36,
        "enable_read_width": 33,
        "enable_write_phase": 0,
        "enable_read_phase": -44,
        "bitmsg_channel": "N0NNR1NNRN",
        "bitmsg_enable": "NWNNEWNNEW",
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
            "x": 0,
            "y": 0,
            "threshold_enforce": 0.3,
        }
    )
    nm.setup_scope_bert(b, measurement_settings)
    save_dict = nm.run_measurement(b, measurement_settings, plot=True)
    b.properties["measurement_settings"] = measurement_settings

    file_path, time_str = qf.save(
        b.properties, measurement_settings["measurement_name"], save_dict
    )
    save_dict["time_str"] = time_str

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

  
    bit_error_rate = save_dict["write_1_read_0"][0] / NUM_MEAS

    save_dict["bit_error_rate"] = bit_error_rate
    nm.write_dict_to_file(file_path, measurement_settings)
    print(f"Bit error rate: {bit_error_rate:.2e}")
    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")