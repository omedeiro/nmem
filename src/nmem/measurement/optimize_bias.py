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
    plot_objective_2D,
)
from skopt.space import Real

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
    print(x)
    print(f"Write Current: {meas_dict['write_current']*1e6:.3f}")
    print(f"Read Current: {meas_dict['read_current']*1e6:.3f}")
    print(f"Enable Write Current: {meas_dict['enable_write_current']*1e6:.3f}")
    print(f"Enable Read Current: {meas_dict['enable_read_current']*1e6:.3f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=False)

    qf.save(b.properties, measurement_name, data_dict)

    errors = data_dict["write_1_read_0"][0] + data_dict["write_0_read_1"][0]

    res = errors / (NUM_MEAS * 2)
    print(res)
    return res


def run_optimize(meas_dict: dict):
    space = [
        Real(10, 220, name="write_current"),
        Real(400, 750, name="read_current"),
        Real(120, 330, name="enable_write_current"),
        Real(100, 280, name="enable_read_current"),
    ]

    nm.setup_scope_bert(b, meas_dict)
    opt_result = gp_minimize(
        partial(objective_bias, meas_dict=meas_dict),
        space,
        n_calls=NUM_CALLS,
        verbose=True,
        x0=meas_dict["opt_x0"],
    )

    return opt_result, meas_dict


if __name__ == "__main__":
    t1 = time.time()

    measurement_name = "nMem_optimize_bias"
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
        "bitmsg_channel": "N0RNR1RNRN",
        "bitmsg_enable": "NWNWEWNWEN",
    }

    current_settings = {
        "write_current": 26e-6,
        "read_current": 617e-6,
        "enable_write_current": 289e-6,
        "enable_read_current": 219e-6,
    }

    opt_x0 = [current * 1e6 for current in current_settings.values()]

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
            "opt_x0": opt_x0,
        }
    )
    # zero_bitidx = 1
    # one_bitidx = 5

    # measurement_settings["bitmsg_channel"] = nm.replace_bit(
    #     measurement_settings["bitmsg_channel"], zero_bitidx, "0"
    # )
    # measurement_settings["bitmsg_channel"] = nm.replace_bit(
    #     measurement_settings["bitmsg_channel"], one_bitidx, "1"
    # )
    # measurement_settings["bitmsg_enable"] = nm.replace_bit(
    #     measurement_settings["bitmsg_enable"], zero_bitidx, "W"
    # )
    # measurement_settings["bitmsg_enable"] = nm.replace_bit(
    #     measurement_settings["bitmsg_enable"], one_bitidx, "W"
    # )

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
