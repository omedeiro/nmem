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
from skopt.plots import (
    plot_convergence,
    plot_evaluations,
    plot_objective,
    plot_objective_2D,
)
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


def objective_write(x, meas_dict: dict):
    meas_dict["write_current"] = x[0] * 1e-6
    meas_dict["enable_write_current"] = x[1] * 1e-6
    print(x)
    print(f"Write Current: {meas_dict['write_current']*1e6:.3f}")
    print(f"Enable Write Current: {meas_dict['enable_write_current']*1e6:.3f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=False)

    qf.save(b.properties, meas_dict["measurement_name"], data_dict)

    errors = data_dict["write_1_read_0"][0] + data_dict["write_0_read_1"][0]

    res = errors / (NUM_MEAS * 2)
    if res == 0.5:
        res = 1
    print(res)
    return res


def run_optimize(meas_dict: dict):
    space = [
        Real(10, 240, name="write_current"),
        Real(120, 330, name="enable_write_current"),
    ]

    nm.setup_scope_bert(b, meas_dict)
    opt_result = gp_minimize(
        partial(objective_write, meas_dict=meas_dict),
        space,
        n_calls=NUM_CALLS,
        verbose=True,
        x0=[37, 290],
    )

    return opt_result, meas_dict


if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_optimize_write"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)

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
        "bitmsg_enable": "NWNNEWNNEN",
    }

    current_settings = {
        "write_current": 37.873e-6,
        "read_current": 619.383e-6,
        "enable_write_current": 290.221e-6,
        "enable_read_current": 209.704e-6,
    }


    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }
    NUM_MEAS = 5000
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
        }
    )

    opt_res, measurement_settings = run_optimize(measurement_settings)
    file_path, time_str = qf.save(
        b.properties, measurement_settings["measurement_name"]
    )

    plot_convergence(opt_res)
    plt.savefig(
        file_path + f"{measurement_settings['measurement_name']}_convergence.png"
    )
    plot_evaluations(opt_res)
    plt.savefig(
        file_path + f"{measurement_settings['measurement_name']}_evaluations.png"
    )
    plot_objective(opt_res)
    plt.savefig(file_path + f"{measurement_settings['measurement_name']}_objective.png")
    print(f"optimal parameters: {opt_res.x}")
    print(f"optimal function value: {opt_res.fun}")

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    nm.write_dict_to_file(file_path, measurement_settings)
    nm.write_dict_to_file(file_path, dict(opt_res))

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
