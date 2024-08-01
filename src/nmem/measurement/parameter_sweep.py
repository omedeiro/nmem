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

plt.rcParams["figure.figsize"] = [10, 12]

CONFIG = r"SPG806_config_ICE.yml"

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
FREQ_IDX = 1
REAL_TIME = 1


if __name__ == "__main__":
    b = nt.nTron(CONFIG)

    if REAL_TIME == 0:
        b.inst.scope.set_sample_mode("Sequence")
        b.inst.scope.set_segments(NUM_MEAS)
    else:
        b.inst.scope.set_sample_mode("Realtime")

    sample_name = [
        b.sample_name,
        b.device_type,
        b.device_name,
        b.properties["Save File"]["cell"],
    ]
    sample_name = str("-".join(sample_name))
    date_str = time.strftime("%Y%m%d")
    measurement_name = f"{date_str}_nMem_ICE_ber"

    measurement_settings = {
        "measurement_name": measurement_name,
        "sample_name": sample_name,
        "write_current": 230e-6,
        "enable_write_current": 350e-6,
        "read_current": 450e-6,  # 1
        "enable_read_current": 285e-6,  # 2
        "threshold_bert": 0.15,
        "num_meas": NUM_MEAS,
        "threshold_read": 100e-3,
        "threshold_enab": 15e-3,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "sample_time": HORIZONTAL_SCALE[FREQ_IDX] * 10,  # 10 divisions
        "num_samples_scope": 5e3,
        "scope_sample_rate": 5e3 / (HORIZONTAL_SCALE[FREQ_IDX] * 10),
        "realtime": REAL_TIME,
        "x": 0,
        "y": 0,
        "num_samples": 2**8,
        "write_width": 100,
        "read_width": 100,  #
        "enable_write_width": 30,
        "enable_read_width": 30,
        "enable_write_phase": 0,
        "enable_read_phase": 30,
        "bitmsg_channel": "NNN0RNN1RN",
        "bitmsg_enable": "NNNWENNWEE",
    }

    t1 = time.time()
    parameter_x = "enable_write_current"
    measurement_settings["x"] = [350e-6]
    # measurement_settings["x"] = np.linspace(285e-6, 350e-6, 11)

    parameter_y = "read_current"
    # measurement_settings["y"] = [300e-6]
    measurement_settings["y"] = np.linspace(440e-6, 500e-6, 11)

    b, measurement_settings, save_dict = nm.run_sweep(
        b, measurement_settings, parameter_x, parameter_y, plot_measurement=True
    )
    file_path, time_str = qf.save(b.properties, measurement_name, save_dict)
    save_dict["time_str"] = time_str
    nm.plot_ber_sweep(
        save_dict,
        measurement_settings,
        file_path,
        parameter_x,
        parameter_y,
        "ber",
    )
    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    nm.write_dict_to_file(file_path, measurement_settings)
