# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:17:34 2023

@author: omedeiro
"""

import sys
import time

import numpy as np
import qnnpy.functions.functions as qf
import qnnpy.functions.ntron as nt
from matplotlib import pyplot as plt

import nmem.measurement.functions as nm

sys.path.append(r"S:\SC\Measurements\SPG806")
config = r"S:\SC\Measurements\SPG806\SPG806_config_ICE.yml"


plt.rcParams["figure.figsize"] = [10, 12]

hor_scale_dict = {
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
# b.inst.scope.recall_setup(1)
# h = [5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
sample_rate_dict = {
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

b = nt.nTron(config)
##############################
sample_name = [
    b.sample_name,
    b.device_type,
    b.device_name,
    b.properties["Save File"]["cell"],
]
sample_name = str("-".join(sample_name))

measurement_name = "nMem_ICE_ber"


realtime = 1
num_meas = 1000

if realtime == 0:
    b.inst.scope.set_sample_mode("Sequence")
    b.inst.scope.set_segments(num_meas)
else:
    b.inst.scope.set_sample_mode("Realtime")

save_dict = {}

FREQ_IDX = 4

measurement_settings = {
    "measurement_name": measurement_name,
    "sample_name": sample_name,
    "write_current": 205e-6,
    "read_current": 590e-6,
    "enable_voltage": 0.0,
    "enable_write_current": 332e-6,
    "enable_read_current": 250e-6,
    "channel_voltage": 0.0,
    "channel_voltage_read": 0.0,
    "wr_ratio": 0.438,
    "ewr_ratio": 1.56,
    "num_meas": num_meas,
    "threshold_read": 100e-3,
    "threshold_enab": 15e-3,
    "sample_rate": sample_rate_dict[FREQ_IDX],
    "hor_scale": hor_scale_dict[FREQ_IDX],
    "num_samples_scope": 5e3,
    "realtime": realtime,
    "x": 0,
    "y": 0,
    "num_samples": 2**8,
    "write_width": 100,
    "read_width": 100,  #
    "enable_write_width": 30,
    "enable_read_width": 30,
    "enable_write_phase": 0,
    "enable_read_phase": 30,
    "bitmsg_channel": "N0R1R1R0RN",
    "bitmsg_enable": "NWNNEWNNEN",
}


t1 = time.time()

#  Write sweep
run1 = False

if run1 == True:
    measurement_settings["x"] = np.array([332e-6])  # np.linspace(325e-6, 335e-6, 21)
    measurement_settings["y"] = np.linspace(185e-6, 225e-6, 11)

    b, measurement_settings, save_dict = nm.run_write_sweep(b, measurement_settings)

    file_path, time_str = qf.save(b.properties, measurement_name, save_dict)
    save_dict["time_str"] = time_str
    nm.plot_ber_sweep(
        save_dict,
        measurement_settings,
        file_path,
        "enable_write_current",
        "write_current",
        "ber",
    )


if run1 == False:
    measurement_settings["x"] = np.array([250e-6])
    measurement_settings["y"] = np.linspace(530e-6, 630e-6, 11)

    b, measurement_settings, save_dict = nm.run_read_sweep(b, measurement_settings)
    file_path, time_str = qf.save(b.properties, measurement_name, save_dict)
    save_dict["time_str"] = time_str
    nm.plot_ber_sweep(
        save_dict,
        measurement_settings,
        file_path,
        "enable_read_current",
        "read_current",
        "ber",
    )


t2 = time.time()
print(f"run time {(t2-t1)/60:.2f} minutes")
b.inst.scope.save_screenshot(
    f"{file_path}_scope_screenshot.png", white_background=False
)
b.inst.awg.set_output(False, 1)
b.inst.awg.set_output(False, 2)

with open(f"{file_path}_measurement_settings.txt", "w") as file:
    for key, value in save_dict.items():
        file.write(f"{key}: {value}\n")
