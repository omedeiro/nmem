# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:17:34 2023

@author: omedeiro
"""

import time
from functools import partial

import qnnpy.functions.functions as qf
from matplotlib import pyplot as plt
from skopt import gp_minimize
from skopt.plots import (
    plot_convergence,
    plot_evaluations,
    plot_objective,
)

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

from nmem.measurement.optimize import objective, optimize_bias


def run_optimize(meas_dict: dict):
    # meas_dict, space, x0 = optimize_all(meas_dict)
    # meas_dict, space, x0 = optimize_bias_phase(meas_dict)
    meas_dict, space, x0, b = optimize_bias(meas_dict)
    # meas_dict, space, x0 = optimize_fixed_write(meas_dict)
    # meas_dict, space, x0, b = optimize_phase(meas_dict)
    # meas_dict, space, x0, b = optimize_read_pulse(meas_dict)

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
        "write_width": 0,
        "read_width": 7,
        "enable_write_width": 3,
        "enable_read_width": 6,
        "enable_write_phase": 1,
        "enable_read_phase": -8,
        "bitmsg_channel": "NN0NRNN1NR",
        "bitmsg_enable": "NNWNENNWNE",
        "threshold_bert": 0.33,
        "threshold_enforced": 0.33,
    }

    current_settings = {
        "write_current": 60e-6,
        "read_current": 630e-6,
        "enable_write_current": 335e-6,
        "enable_read_current": 236e-6,
    }

    NUM_MEAS = 2000
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

    nm.setup_scope_bert(
        b, measurement_settings, division_zero=(3.5, 4.0), division_one=(5.5, 6.0)
    )

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
