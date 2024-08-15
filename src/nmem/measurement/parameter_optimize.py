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
import scipy.optimize as opt
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

NUM_MEAS = 1000
NUM_DIVISIONS = 10
NUM_SAMPLES = 5e3
NUM_POINTS = 2**8
ENABLE_WIDTH = 100e-9
BIAS_WIDTH = 300e-9


def optimize_bias_currents(x, meas_dict: dict):
    meas_dict["write_current"] = x[0]
    meas_dict["read_current"] = x[1]
    print(x)
    data_dict = nm.run_measurement(b, meas_dict, plot=True)
    errors = data_dict["write_1_read_0"] + data_dict["write_0_read_1"]
    print(errors)
    return errors


def optimize_read_widths(x, meas_dict: dict):
    meas_dict["read_width"] = x[0]
    meas_dict["enable_read_width"] = x[1]
    meas_dict["enable_read_phase"] = x[2]
    print(x)
    data_dict = nm.run_measurement(b, meas_dict, plot=True)
    errors = data_dict["write_1_read_0"] + data_dict["write_0_read_1"]
    print(errors)
    return errors


def optimize_write_widths(x, meas_dict: dict):
    meas_dict["write_width"] = x[0]
    meas_dict["enable_write_width"] = x[1]
    meas_dict["enable_write_phase"] = x[2]
    print(x)
    data_dict = nm.run_measurement(b, meas_dict, plot=True)
    errors = data_dict["write_1_read_0"] + data_dict["write_0_read_1"]
    print(errors)
    return errors


def optimize_pulse_phase(x, meas_dict: dict):
    meas_dict["enable_write_phase"] = x[0]
    meas_dict["enable_read_phase"] = x[1]
    print(x)
    data_dict = nm.run_measurement(b, meas_dict, plot=True)
    errors = data_dict["write_1_read_0"] + data_dict["write_0_read_1"]
    print(errors)
    return errors


def optimize_read(x, meas_dict: dict):
    meas_dict["read_current"] = x[0] * meas_dict["opt_scale"][0]
    meas_dict["enable_read_current"] = x[1] * meas_dict["opt_scale"][1]
    print(f"Read Current: {meas_dict['read_current']*1e6:.2f}")
    print(f"Enable Read Current: {meas_dict['enable_read_current']*1e6:.2f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=True)
    errors = data_dict["write_1_read_0"] + data_dict["write_0_read_1"]
    print(errors, (errors / (NUM_MEAS * 2)))
    return errors / (NUM_MEAS * 2)


def optimize_write(x, meas_dict: dict):
    meas_dict["write_current"] = x[0] * meas_dict["opt_scale"][0]
    meas_dict["enable_write_current"] = x[1] * meas_dict["opt_scale"][1]
    print(x)
    data_dict = nm.run_measurement(b, meas_dict, plot=True)
    errors = data_dict["write_1_read_0"] + data_dict["write_0_read_1"]
    print(errors)
    return errors / (NUM_MEAS * 2)


def run_optimize(meas_dict: dict):
    # x0 = np.array([75.9e-6, 375e-6, 355e-6, 295e-6])
    x0 = np.array([75.9e-6, 375e-6])

    # x0_pulse = np.array([90, 90, 30, 30, 0, 0])
    # bounds_pulse = [(50, 150), (50, 150), (20, 60), (20, 60), (-20, 20), (-20, 20)]

    # opt_result = opt.minimize(
    #     optimize_bias_currents,
    #     x0=x0,
    #     args=(meas_dict),
    #     bounds=[(0, 100e-6), (250e-6, 400e-6)],
    #     options={"disp": True, "maxiter": 10},
    #     # method="Nelder-Mead",
    #     method="L-BFGS-B",
    # )

    x0_read = np.array([375e-6, 295e-6])
    bounds_read = [(200e-6, 500e-6), (250e-6, 400e-6)]
    meas_dict["opt_scale"] = [500e-6, 400e-6]
    x0_read = x0_read / meas_dict["opt_scale"]
    bounds_read = [(200e-6 / 500e-6, 1), (250e-6 / 400e-6, 1)]
    opt_result = opt.minimize(
        optimize_read,
        x0=x0_read,
        args=(meas_dict),
        bounds=bounds_read,
        options={"disp": True, "maxiter": 10},
        # method="Nelder-Mead",
        method="L-BFGS-B",
    )

    # x0_write = np.array([75.9e-6, 355e-6])
    # bounds_write = [(10e-6, 100e-6), (250e-6, 400e-6)]

    # opt_result = opt.minimize(
    #     optimize_write,
    #     x0=x0_write,
    #     args=(meas_dict),
    #     bounds=bounds_write,
    #     options={"disp": True, "maxiter": 10},
    #     # method="Nelder-Mead",
    #     method="L-BFGS-B",
    # )

    # x0_phase = np.array([0, 30])
    # bounds_phase = [(-50, 50), (-50, 50)]

    # opt_result = opt.minimize(
    #     optimize_pulse_phase,
    #     x0=x0_phase,
    #     args=(meas_dict),
    #     bounds=bounds_phase,
    #     options={"disp": True, "maxiter": 10},
    #     # method="Nelder-Mead",
    #     method="L-BFGS-B",
    # )

    # x0_write_width = np.array([90, 30, 0])
    # bounds_write_width = [(50, 150), (20, 60), (-20, 20)]

    # opt_result = opt.minimize(
    #     optimize_write_widths,
    #     x0=x0_write_width,
    #     args=(meas_dict),
    #     bounds=bounds_write_width,
    #     options={"disp": True, "maxiter": 10},
    #     # method="Nelder-Mead",
    #     method="L-BFGS-B",
    # )
    return opt_result


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
        "write_width": 90,
        "read_width": 90,  #
        "enable_write_width": 30,
        "enable_read_width": 30,
        "enable_write_phase": 0,
        "enable_read_phase": 0,
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
        "write_current": 80.1e-6,
        "read_current": 408e-6,
        "enable_write_current": 354.5e-6,
        "enable_read_current": 286e-6,
        "threshold_bert": 0.15,
        "bitmsg_channel": "N0RNR1NRRN",
        "bitmsg_enable": "NWNWEWWNEW",
    }

    parameter_x = "enable_read_current"
    measurement_settings["x"] = np.array([295e-6])
    # measurement_settings["x"] = np.linspace(150e-6, 250e-6, 9)

    optimization = run_optimize(measurement_settings)
    print(f"Optimization Result: {optimization.x}")
    b.properties["measurement_settings"] = measurement_settings

    # file_path, time_str = qf.save(b.properties, measurement_name, save_dict)
    # save_dict["time_str"] = time_str
    # nm.plot_ber_sweep(
    #     save_dict,
    #     measurement_settings,
    #     file_path,
    #     parameter_x,
    #     parameter_y,
    #     "bit_error_rate",
    # )

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    # nm.write_dict_to_file(file_path, measurement_settings)

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
