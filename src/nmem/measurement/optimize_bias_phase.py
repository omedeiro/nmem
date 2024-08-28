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


def objective_bias(x, meas_dict: dict):
    meas_dict["write_current"] = x[0] * 1e-6
    meas_dict["read_current"] = x[1] * 1e-6
    meas_dict["enable_write_current"] = x[2] * 1e-6
    meas_dict["enable_read_current"] = x[3] * 1e-6
    meas_dict["enable_write_phase"] = x[4]
    meas_dict["enable_read_phase"] = x[5]
    print(x)
    print(f"Write Current: {meas_dict['write_current']*1e6:.3f}")
    print(f"Read Current: {meas_dict['read_current']*1e6:.3f}")
    print(f"Enable Write Current: {meas_dict['enable_write_current']*1e6:.3f}")
    print(f"Enable Read Current: {meas_dict['enable_read_current']*1e6:.3f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=False)

    qf.save(b.properties, measurement_name, data_dict)

    errors = data_dict["write_1_read_0"][0] + data_dict["write_0_read_1"][0]

    res = errors / (NUM_MEAS * 2)
    if res == 0.5:
        res = 1
    # elif res > 0.5:
    #     res = 1 - res

    print(res)
    return res


def run_optimize(meas_dict: dict):
    space = [
        Real(10, 240, name="write_current"),
        Real(420, 800, name="read_current"),
        Real(120, 380, name="enable_write_current"),
        Real(100, 280, name="enable_read_current"),
        Integer(-50, 40, name="enable_write_phase"),
        Integer(-50, 40, name="enable_read_phase"),
    ]

    nm.setup_scope_bert(b, meas_dict)
    opt_result = gp_minimize(
        partial(objective_bias, meas_dict=meas_dict),
        space,
        n_calls=NUM_CALLS,
        verbose=True,
        x0=[36.935, 638.626, 291.405, 208.334, -12, 30],
    )

    return opt_result, meas_dict


if __name__ == "__main__":
    t1 = time.time()

    measurement_name = "nMem_optimize_bias_phase"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 90,
        "read_width": 82,  #
        "enable_write_width": 36,
        "enable_read_width": 33,
        "enable_write_phase": -12,
        "enable_read_phase": 30,
        "bitmsg_channel": "N0RNR1RNRN",
        "bitmsg_enable": "NWNWEWNWEW",
    }

    current_settings = {
        "write_current": 36.935e-6,
        "read_current": 638.626e-6,
        "enable_write_current": 291.405e-6,
        "enable_read_current": 208.334e-6,
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }
    NUM_MEAS = 1000
    NUM_CALLS = 100
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
    plt.savefig(f"{file_path}_convergence.png", dpi=300, format="png")
    plt.show()
    plot_evaluations(opt_res)
    plt.savefig(f"{file_path}_evaluations.png")
    plt.show()
    plot_objective(opt_res)
    plt.savefig(f"{file_path}_objective.png")
    plt.show()
    plot_objective_2D(opt_res, "enable_write_current", "write_current")
    plt.savefig(f"{file_path}_objective_2D_write.png")
    plt.show()
    plot_objective_2D(opt_res, "enable_read_current", "read_current")
    plt.savefig(f"{file_path}_objective_2D_read.png", format="png")
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