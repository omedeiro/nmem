# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:17:34 2023

@author: omedeiro
"""

import time
from functools import partial

import numpy as np
import qnnpy.functions.functions as qf
import qnnpy.functions.ntron as nt
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from skopt.space import Integer, Real

import nmem.measurement.functions as nm
from nmem.measurement.cells import (
    CELLS,
    CONFIG,
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


def objective_waveform(x, meas_dict: dict):
    meas_dict["write_width"] = x[0]
    meas_dict["read_width"] = x[1]
    meas_dict["enable_write_width"] = x[2]
    meas_dict["enable_read_width"] = x[3]
    meas_dict["enable_write_phase"] = x[4]
    meas_dict["enable_read_phase"] = x[5]
    print(x)
    print(f"Write Width: {meas_dict['write_width']:.2f}")
    print(f"Read Width: {meas_dict['read_width']:.2f}")
    print(f"Enable Write Width: {meas_dict['enable_write_width']:.2f}")
    print(f"Enable Read Width: {meas_dict['enable_read_width']:.2f}")
    print(f"Enable Write Phase: {meas_dict['enable_write_phase']:.2f}")
    print(f"Enable Read Phase: {meas_dict['enable_read_phase']:.2f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=False)

    qf.save(b.properties, measurement_name, data_dict)

    errors = data_dict["write_1_read_0"][0] + data_dict["write_0_read_1"][0]
    res = errors / (NUM_MEAS * 2)
    print(res)
    return errors / (NUM_MEAS * 2)


def run_optimize(meas_dict: dict):
    space = [
        Integer(50, 100, name="write_width"),
        Integer(10, 100, name="read_width"),
        Integer(5, 60, name="enable_write_width"),
        Integer(5, 60, name="enable_read_width"),
        Integer(-40, 20, name="enable_write_phase"),
        Integer(-40, -10, name="enable_read_phase"),
    ]
    nm.setup_scope_bert(b, meas_dict)
    opt_result = gp_minimize(
        partial(objective_waveform, meas_dict=meas_dict),
        space,
        n_calls=100,
        verbose=True,
        x0=meas_dict["opt_x0"],
    )

    return opt_result, meas_dict


if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_optimize_waveform"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    waveform_settings = {
        "write_width": 90,
        "read_width": 82,  #
        "enable_write_width": 36,
        "enable_read_width": 33,
        "enable_write_phase": -12,
        "enable_read_phase": -35,
    }
    opt_x0 = list(waveform_settings.values())

    current_settings = {
        "write_current": 121.7e-6,
        "read_current": 596.7e-6,
        "enable_write_current": 296.2e-6,
        "enable_read_current": 202.4e-6,
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }
    NUM_MEAS = 100

    measurement_settings.update(
        {
            **waveform_settings,
            **current_settings,
            **scope_settings,
            "CELLS": CELLS,
            "HEATERS": HEATERS,
            "num_meas": NUM_MEAS,
            "spice_device_current": SPICE_DEVICE_CURRENT,
            "x": 0,
            "y": 0,
            "threshold_bert": 0.2,
            "num_points": NUM_POINTS,
            "sample_rate": SAMPLE_RATE[FREQ_IDX],
            "bitmsg_channel": "N0RNR1RNRN",
            "bitmsg_enable": "NWNWEWNWEW",
            "opt_x0": opt_x0,
        }
    )

    opt_res, measurement_settings = run_optimize(measurement_settings)
    file_path, time_str = qf.save(b.properties, measurement_name)

    plot_convergence(opt_res)
    plt.savefig(file_path + "{measurement_name}_convergence.png")
    plot_evaluations(opt_res)
    plt.savefig(file_path + "{measurement_name}_evaluations.png")
    plot_objective(opt_res)
    plt.savefig(file_path + "{measurement_name}_objective.png")
    print(f"optimal parameters: {opt_res.x}")
    print(f"optimal function value: {opt_res.fun}")

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    nm.write_dict_to_file(file_path, measurement_settings)
    nm.write_dict_to_file(file_path, dict(opt_res))

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
