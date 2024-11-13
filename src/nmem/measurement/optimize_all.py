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
from nmem.measurement.optimize import (
    objective,
    optimize_all,
    optimize_bias,
    optimize_enable,
    optimize_fixed_write,
    optimize_phase,
    optimize_read_pulse,
    optimize_read, 
    optimize_write
)

plt.rcParams["figure.figsize"] = [10, 12]


def run_optimize(meas_dict: dict):
    # meas_dict, space, x0 = optimize_all(meas_dict)
    meas_dict, space, x0, b = optimize_bias(meas_dict)
    # meas_dict, space, x0, b = optimize_read(meas_dict)
    # meas_dict, space, x0 = optimize_fixed_write(meas_dict)
    # meas_dict, space, x0, b = optimize_phase(meas_dict)
    # meas_dict, space, x0, b = optimize_read_pulse(meas_dict)
    # meas_dict, space, x0, b = optimize_write(meas_dict)

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

    slow_write = {
        "write_width": 40,
        "enable_write_width": 2,
        "enable_write_phase": -7,
    }
    fast_write = {
        "write_width": 0,
        "enable_write_width": 2,
        "enable_write_phase": -7,
    }
    slow_read = {
        "read_width": 40,
        "enable_read_width": 4,
        "enable_read_phase": -7,
    }
    fast_read = {
        "read_width": 8,
        "enable_read_width": 4,
        "enable_read_phase": -8,
    }

    two_nulls = {
        "bitmsg_channel": "N0NNRN1NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }
    zero_nulls = {
        "bitmsg_channel": "NNN0RNNN1R",
        "bitmsg_enable": "NNNWENNNWE",
    }
    waveform_settings = {   
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        **fast_write,
        **fast_read,
        **two_nulls,
        "threshold_bert": 0.4,
        "threshold_enforced": 0.4,
    }

    current_settings = {
        "write_current": 30e-6,
        "read_current": 645e-6,
        "enable_write_current": 350e-6,
        "enable_read_current": 250e-6,
    }

    NUM_MEAS = 200
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
        b,
        measurement_settings,
        division_zero=(1.9, 2.5),
        division_one=(5.9, 6.5),
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
