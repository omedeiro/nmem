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
    measurement_name = "nMem_delay_error"
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
        "bitmsg_channel": "NNNNNNNNRN",
        "bitmsg_enable": "NNNNNNNNEW",
    }
    current_settings = {
        "write_current": 30e-6,
        "read_current": 637e-6,
        "enable_write_current": 288e-6,
        "enable_read_current": 218e-6,
    }

    i = 2
    waveform_settings["bitmsg_channel"] = nm.replace_bit(
        waveform_settings["bitmsg_channel"], i, "0"
    )
    waveform_settings["bitmsg_enable"] = nm.replace_bit(
        waveform_settings["bitmsg_enable"], i, "W"
    )

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
