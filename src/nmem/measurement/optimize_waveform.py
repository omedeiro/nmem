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
from skopt.plots import plot_convergence, plot_evaluations, plot_objective, plot_objective_2D
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
        Integer(5, 70, name="write_width"),
        Integer(5, 70, name="read_width"),
        Integer(5, 70, name="enable_write_width"),
        Integer(5, 70, name="enable_read_width"),
        Integer(-50, 50, name="enable_write_phase"),
        Integer(-50, 50, name="enable_read_phase"),
    ]
    nm.setup_scope_bert(b, meas_dict)
    opt_result = gp_minimize(
        partial(objective_waveform, meas_dict=meas_dict),
        space,
        n_calls=NUM_CALLS,
        verbose=True,
        x0=meas_dict["opt_x0"],
    )

    return opt_result, meas_dict


if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_optimize_waveform"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    waveform_settings = {
        "write_width": 60,
        "read_width": 60,  #
        "enable_write_width": 20,
        "enable_read_width": 20,
        "enable_write_phase": 0,
        "enable_read_phase": -20,
    }
    opt_x0 = list(waveform_settings.values())

    current_settings = {
        "write_current": 68.58e-6,
        "read_current": 435.369e-6,
        "enable_write_current": 330.383e-6,
        "enable_read_current": 270e-6,
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }
    NUM_MEAS = 2000
    NUM_CALLS = 40
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
    file_path, time_str = qf.save(b.properties, measurement_settings["measurement_name"])
    plot_convergence(opt_res)
    plt.savefig(f"{file_path}_convergence.png", dpi=300, format="png")
    plt.show()
    plot_evaluations(opt_res)
    plt.savefig(f"{file_path}_evaluations.png")
    plt.show()
    plot_objective(opt_res)
    plt.savefig(f"{file_path}_objective.png")
    plt.show()
    print(f"optimal parameters: {opt_res.x}")
    for i, x in enumerate(opt_res.x):
        print(f"{x:.3f}")
    print(f"optimal function value: {opt_res.fun}")

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    nm.write_dict_to_file(file_path, measurement_settings)
    nm.write_dict_to_file(file_path, dict(opt_res))

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
