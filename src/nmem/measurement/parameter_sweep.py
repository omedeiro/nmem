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
from nmem.calculations.calculations import htron_critical_current
from nmem.measurement.cells import CELLS

plt.rcParams["figure.figsize"] = [10, 12]

CONFIG = r"SPG806_config_ICE.yml"


FREQ_IDX = 1
HORIZONTAL_SCALE = {
    0: 5e-7,
    1: 1e-6,
    2: 2e-6,
    3: 5e-6,
    4: 1e-5,
    5: 2e-5,
    6: 5e-5,
    7: 1e-4,
    8: 2e-4,
    9: 5e-4,
}
SAMPLE_RATE = {
    0: 512e6,
    1: 256e6,
    2: 128e6,
    3: 512e5,
    4: 256e5,
    5: 128e5,
    6: 512e4,
    7: 256e4,
    8: 128e4,
    9: 512e3,
}

NUM_MEAS = 500
NUM_DIVISIONS = 10
NUM_SAMPLES = 5e3
NUM_POINTS = 2**8
ENABLE_WIDTH = 100e-9
BIAS_WIDTH = 300e-9


if __name__ == "__main__":
    t1 = time.time()
    b = nt.nTron(CONFIG)
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
        "num_points": 256,
        "write_width": 100,
        "read_width": 100,  #
        "enable_write_width": 30,
        "enable_read_width": 30,
        "enable_write_phase": 0,
        "enable_read_phase": 30,
    }

    measurement_settings = {
        **waveform_settings,
        "measurement_name": measurement_name,
        "sample_name": sample_name,
        "CELLS": CELLS,
        "num_meas": NUM_MEAS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "sample_time": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "num_samples_scope": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
        "x": 0,
        "y": 0,
        "write_current": 70e-6,
        "read_current": 463e-6,
        "enable_write_current": 290e-6,
        "enable_read_current": 290e-6,
        "threshold_bert": 0.2,
        "bitmsg_channel": "N0NNR1NNRN",
        "bitmsg_enable": "NWNNEWNNEW",
    }

    parameter_x = "enable_read_current"
    measurement_settings["x"] = np.array([290e-6])
    # measurement_settings["x"] = np.linspace(150e-6, 250e-6, 9)

    read_sweep = False
    if read_sweep:
        parameter_y = "read_current"
        # measurement_settings["y"] = [375e-6]
        read_critical_current = htron_critical_current(
            measurement_settings["enable_read_current"],
            CELLS[current_cell]["slope"],
            CELLS[current_cell]["intercept"] * 1e-6,
        )
        measurement_settings["y"] = np.linspace(
            read_critical_current * 0.85, read_critical_current * 1.1, 11
        )
    else:
        parameter_y = "write_current"
        # measurement_settings["y"] = [120e-6]
        # measurement_settings["y"] = np.linspace(10e-6, 230e-6, 11)
        write_critical_current = htron_critical_current(
            measurement_settings["enable_write_current"],
            CELLS[current_cell]["slope"],
            CELLS[current_cell]["intercept"] * 1e-6,
        )
        measurement_settings["y"] = np.linspace(
            write_critical_current * 0.1, write_critical_current * 0.9, 11
        )

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
