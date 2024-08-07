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


if __name__ == "__main__":
    t1 = time.time()
    b = nt.nTron(CONFIG)

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
        "write_current": 100e-6,
        "read_current": 570e-6,
        "enable_write_current": 282.5e-6,
        "enable_read_current": 230e-6,
        "threshold_bert": 0.2,
        "num_meas": NUM_MEAS,
        "threshold_read": 100e-3,
        "threshold_enab": 15e-3,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "sample_time": HORIZONTAL_SCALE[FREQ_IDX] * 10,  # 10 divisions
        "num_samples_scope": 5e3,
        "scope_sample_rate": 5e3 / (HORIZONTAL_SCALE[FREQ_IDX] * 10),
        "x": 0,
        "y": 0,
        "num_samples": 2**8,
        "write_width": 100,
        "read_width": 100,  #
        "enable_write_width": 30,
        "enable_read_width": 30,
        "enable_write_phase": 0,
        "enable_read_phase": 30,
        # "bitmsg_channel": "NNN0RNN1RN",
        # "bitmsg_enable": "NNNWENNWEE",
        "bitmsg_channel": "N0NNR1NNRN",
        "bitmsg_enable": "NWNNEWNNEE",
    }

    parameter_x = "enable_write_current"
    # measurement_settings["x"] = [240e-6]
    measurement_settings["x"] = np.linspace(320e-6, 400e-6, 11)

    parameter_y = "write_current"
    measurement_settings["y"] = [100e-6]
    # measurement_settings["y"] = np.linspace(0e-6, 200e-6, 21)

    save_dict = nm.run_sweep(
        b, measurement_settings, parameter_x, parameter_y, plot_measurement=False
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
