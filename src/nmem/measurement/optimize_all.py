# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:17:34 2023

@author: omedeiro
"""

import time
from functools import partial
from qnnpy.functions.ntron import nTron
import qnnpy.functions.functions as qf
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.plots import (
    plot_convergence,
    plot_evaluations,
    plot_objective,
)
from skopt.space import Integer, Real
import numpy as np
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


def update_space(meas_dict: dict, space: list, x0: list):
    for i, s in enumerate(space):
        if s.name in meas_dict:
            if meas_dict[s.name] < 1e-3:
                meas_dict[s.name] = x0[i] * 1e-6
            else:
                meas_dict[s.name] = x0[i]
    return meas_dict


def objective_primary(w1r0: np.ndarray, w0r1: np.ndarray):
    errors = w1r0[0] + w0r1[0]
    res = errors / (NUM_MEAS * 2)
    if res == 0.5:
        res = 1
    return res


def optimize_bias_phase(meas_dict: dict):
    measurement_name = "nMem_optimize_bias_phase"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(10, 240, name="write_current"),
        Real(420, 800, name="read_current"),
        Real(120, 380, name="enable_write_current"),
        Real(100, 280, name="enable_read_current"),
        Integer(-50, 40, name="enable_write_phase"),
        Integer(-50, 40, name="enable_read_phase"),
    ]
    x0 = [36.935, 638.626, 291.405, 208.334, -12, 30]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_bias(meas_dict: dict):
    measurement_name = "nMem_optimize_bias"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(10, 220, name="write_current"),
        Real(400, 750, name="read_current"),
        Real(120, 330, name="enable_write_current"),
        Real(100, 280, name="enable_read_current"),
    ]

    x0 = [68.58, 427.135, 330.383, 265.073]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_fixed_write(meas_dict: dict):
    measurement_name = "nMem_optimize_fixed_write"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(420, 660, name="read_current"),
        Real(190, 330, name="enable_write_current"),
        Real(140, 260, name="enable_read_current"),
    ]

    x0 = [605, 240, 148]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_phase(meas_dict: dict):
    measurement_name = "nMem_optimize_phase"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Integer(-60, 50, name="enable_write_phase"),
        Integer(-50, 50, name="enable_read_phase"),
    ]

    x0 = [0, -44]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_read_enable(meas_dict: dict):
    measurement_name = "nMem_optimize_read_enable"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(400, 690, name="read_current"),
        Real(110, 280, name="enable_read_current"),
        Integer(10, 100, name="read_width"),
        Integer(5, 60, name="enable_read_width"),
        Integer(-50, 40, name="enable_read_phase"),
    ]

    x0 = [619, 209, 82, 33, -44]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_read(meas_dict: dict):
    measurement_name = "nMem_optimize_read"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(400, 690, name="read_current"),
        Integer(5, 100, name="read_width"),
    ]

    x0 = [619, 82]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b

def optimize_read_pulse(meas_dict: dict):
    measurement_name = "nMem_optimize_read_pulse"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Integer(5, 35, name="read_width"),
        Integer(-50, 50, name="read_phase"),
    ]

    x0 = [35, -20]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b

def optimize_write(meas_dict: dict):
    measurement_name = "nMem_optimize_write"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(10, 240, name="write_current"),
        Integer(5, 100, name="write_width"),
    ]
    x0 = [68.58, 90]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_write_enable(meas_dict: dict):
    measurement_name = "nMem_optimize_write_enable"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(10, 240, name="write_current"),
        Real(120, 350, name="enable_write_current"),
        Integer(5, 100, name="write_width"),
        Integer(5, 100, name="enable_write_width"),
        Integer(-40, 40, name="enable_write_phase"),
    ]
    x0 = [68.58, 330.383, 90, 30, 10]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_waveform(meas_dict: dict):
    measurement_name = "nMem_optimize_waveform"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Integer(5, 100, name="write_width"),
        Integer(5, 100, name="read_width"),
        Integer(5, 100, name="enable_write_width"),
        Integer(5, 100, name="enable_read_width"),
        Integer(-40, 40, name="enable_write_phase"),
        Integer(-40, 40, name="enable_read_phase"),
    ]

    x0 = [90, 90, 30, 30, 10, -30]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_all(meas_dict: dict):
    measurement_name = "nMem_optimize_all"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(10, 240, name="write_current"),
        Real(420, 800, name="read_current"),
        Real(120, 350, name="enable_write_current"),
        Real(120, 270, name="enable_read_current"),
        Integer(5, 100, name="write_width"),
        Integer(5, 100, name="read_width"),
        Integer(5, 100, name="enable_write_width"),
        Integer(5, 100, name="enable_read_width"),
        Integer(-40, 40, name="enable_write_phase"),
        Integer(-40, 40, name="enable_read_phase"),
    ]
    x0 = [68.58, 427.135, 330.383, 265.073, 90, 90, 30, 30, 10, -30]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def objective(x, meas_dict: dict, space: list, b: nTron):
    print(x)
    meas_dict = update_space(meas_dict, space, x)

    data_dict = nm.run_measurement(b, meas_dict, plot=False)

    qf.save(b.properties, meas_dict["measurement_name"], data_dict)

    return objective_primary(data_dict["write_1_read_0"], data_dict["write_0_read_1"])


def run_optimize(meas_dict: dict):
    # meas_dict, space, x0 = optimize_all(meas_dict)
    # meas_dict, space, x0 = optimize_bias_phase(meas_dict)
    # meas_dict, space, x0 = optimize_bias(meas_dict)
    # meas_dict, space, x0 = optimize_fixed_write(meas_dict)
    # meas_dict, space, x0, b = optimize_phase(meas_dict)
    meas_dict, space, x0, b = optimize_read_pulse(meas_dict)

    opt_result = gp_minimize(
        partial(objective, meas_dict=meas_dict, space=space, b=b),
        space,
        n_calls=NUM_CALLS,
        verbose=True,
        x0=x0,
    )

    return opt_result, meas_dict


if __name__ == "__main__":
    t1 = time.time()

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 30,
        "read_width": 30,  #
        "enable_write_width": 30,
        "enable_read_width": 35,
        "enable_write_phase": 0,
        "enable_read_phase": -20,
        "bitmsg_channel": "N0NNRN1NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }

    current_settings = {
        "write_current": 20.160e-6,
        "read_current": 640.192e-6,
        "enable_write_current": 289.500e-6,
        "enable_read_current": 212.843e-6,
    }

    NUM_MEAS = 1000
    NUM_CALLS = 40
    measurement_settings = {
        **waveform_settings,
        **current_settings,
        **DEFAULT_SCOPE,
        "CELLS": CELLS,
        "HEATERS": HEATERS,
        "num_meas": NUM_MEAS,
        "spice_device_current": SPICE_DEVICE_CURRENT,
        "x": 0,
        "y": 0,
        # "threshold_bert": 0.2,
    }

    meas_dict, b = nm.initilize_measurement(CONFIG, "nMem_optimize")

    nm.setup_scope_bert(b, measurement_settings)

    opt_res, measurement_settings = run_optimize(measurement_settings)

    measurement_name = measurement_settings["measurement_name"]
    file_path, time_str = qf.save(b.properties, measurement_name)

    plot_convergence(opt_res)
    plt.savefig(file_path + f"{measurement_name}_convergence.png")
    plot_evaluations(opt_res)
    plt.savefig(file_path + f"{measurement_name}_evaluations.png")
    plot_objective(opt_res)
    plt.savefig(file_path + f"{measurement_name}_objective.png")
    print(f"optimal parameters: {opt_res.x}")
    print(f"optimal function value: {opt_res.fun}")

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    nm.write_dict_to_file(file_path, measurement_settings)
    nm.write_dict_to_file(file_path, dict(opt_res))

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
