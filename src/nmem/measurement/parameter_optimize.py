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

CONFIG = r"SPG806_config_ICE.yml"


def objective_converse(x, meas_dict: dict):
    meas_dict["write_current"] = x[0] * 1e-6
    meas_dict["enable_read_current"] = x[1] * 1e-6
    print(x)
    print(f"Write Current: {meas_dict['write_current']*1e6:.2f}")
    print(f"Enable Read Current: {meas_dict['enable_read_current']*1e6:.2f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=True)

    qf.save(b.properties, measurement_name, data_dict)

    errors = data_dict["write_1_read_0"] + data_dict["write_0_read_1"]
    print(errors / (NUM_MEAS * 2))
    return errors / (NUM_MEAS * 2)


def objective_bias(x, meas_dict: dict):
    meas_dict["write_current"] = x[0] * 1e-6
    meas_dict["read_current"] = x[1] * 1e-6
    meas_dict["enable_write_current"] = x[2] * 1e-6
    meas_dict["enable_read_current"] = x[3] * 1e-6
    print(x)
    print(f"Write Current: {meas_dict['write_current']*1e6:.2f}")
    print(f"Read Current: {meas_dict['read_current']*1e6:.2f}")
    print(f"Enable Write Current: {meas_dict['enable_write_current']*1e6:.2f}")
    print(f"Enable Read Current: {meas_dict['enable_read_current']*1e6:.2f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=True)

    qf.save(b.properties, measurement_name, data_dict)

    errors = data_dict["write_1_read_0"][0] + data_dict["write_0_read_1"][0]
    # print(np.abs((errors / (NUM_MEAS * 2)) - 1))

    res = errors / (NUM_MEAS * 2)
    # if res > 0.5:
    # res = 1 - res
    print(res)
    return res


def objective_fixed_write(x, meas_dict: dict):
    meas_dict["read_current"] = x[0] * 1e-6
    meas_dict["enable_write_current"] = x[1] * 1e-6
    meas_dict["enable_read_current"] = x[2] * 1e-6
    print(x)
    print(f"Read Current: {meas_dict['read_current']*1e6:.2f}")
    print(f"Enable Write Current: {meas_dict['enable_write_current']*1e6:.2f}")
    print(f"Enable Read Current: {meas_dict['enable_read_current']*1e6:.2f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=True)

    qf.save(b.properties, measurement_name, data_dict)

    errors = data_dict["write_1_read_0"] + data_dict["write_0_read_1"]
    # print(np.abs((errors / (NUM_MEAS * 2)) - 1))
    print(errors / (NUM_MEAS * 2))
    return errors / (NUM_MEAS * 2)


def objective_read(x, meas_dict: dict):
    meas_dict["read_current"] = x[0] * 1e-6
    meas_dict["enable_read_current"] = x[1] * 1e-6
    print(x)
    print(f"Write Current: {meas_dict['write_current']*1e6:.2f}")
    print(f"Read Current: {meas_dict['read_current']*1e6:.2f}")
    print(f"Enable Write Current: {meas_dict['enable_write_current']*1e6:.2f}")
    print(f"Enable Read Current: {meas_dict['enable_read_current']*1e6:.2f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=True)

    qf.save(b.properties, measurement_name, data_dict)

    errors = data_dict["write_1_read_0"][0] + data_dict["write_0_read_1"][0]
    print(errors / (NUM_MEAS * 2))
    return errors / (NUM_MEAS * 2)


def objective_write(x, meas_dict: dict):
    meas_dict["write_current"] = x[0] * 1e-6
    meas_dict["enable_write_current"] = x[1] * 1e-6
    print(x)
    print(f"Write Current: {meas_dict['write_current']*1e6:.2f}")
    print(f"Enable Write Current: {meas_dict['enable_write_current']*1e6:.2f}")
    data_dict = nm.run_measurement(b, meas_dict, plot=True)

    qf.save(b.properties, measurement_name, data_dict)

    errors = data_dict["write_1_read_0"][0] + data_dict["write_0_read_1"][0]

    res = errors / (NUM_MEAS * 2)
    # if res > 0.5:
    #     res = 1 - res
    print(res)
    return res


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
    print(errors / (NUM_MEAS * 2))
    return errors / (NUM_MEAS * 2)


def run_optimize(meas_dict: dict):
    space = [
        Real(40, 90, name="write_current"),
        # Real(650, 720, name="read_current"),
        Real(200, 255, name="enable_write_current"),
        # Real(170, 190, name="enable_read_current"),
    ]
    # space = [
    #     Integer(60, 100, name="write_width"),
    #     Integer(10, 100, name="read_width"),
    #     Integer(10, 50, name="enable_write_width"),
    #     Integer(10, 50, name="enable_read_width"),
    #     Integer(-40, 20, name="enable_write_phase"),
    #     Integer(-40, -10, name="enable_read_phase"),
    # ]
    nm.setup_scope_bert(b, meas_dict)
    opt_result = gp_minimize(
        partial(objective_write, meas_dict=meas_dict),
        space,
        n_calls=20,
        verbose=True,
        x0=[
            49,
            228,
        ],
        # x0 = [
        #     90,
        #     82,
        #     36,
        #     33,
        #     -12,
        #     -35,
        # ]
    )

    return opt_result, meas_dict


if __name__ == "__main__":
    t1 = time.time()
    b = nt.nTron(CONFIG)

    b.inst.awg.write("SOURce1:FUNCtion:ARBitrary:FILTer OFF")
    b.inst.awg.write("SOURce2:FUNCtion:ARBitrary:FILTer OFF")

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
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 90,
        "read_width": 82,  #
        "enable_write_width": 36,
        "enable_read_width": 33,
        "enable_write_phase": -12,
        "enable_read_phase": -35,
        "bitmsg_channel": "N0RNR1RNRN",
        "bitmsg_enable": "NWNWEWNWEW",
    }

    current_settings = {
        "write_current": 45.213e-6,
        "read_current": 675.9e-6,
        "enable_write_current": 224.494e-6,
        "enable_read_current": 179.9e-6,
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }
    NUM_MEAS = 500

    measurement_settings = {
        **waveform_settings,
        **current_settings,
        **scope_settings,
        "measurement_name": measurement_name,
        "sample_name": sample_name,
        "cell": current_cell,
        "CELLS": CELLS,
        "HEATERS": HEATERS,
        "num_meas": NUM_MEAS,
        "spice_device_current": SPICE_DEVICE_CURRENT,
        "x": 0,
        "y": 0,
        "threshold_bert": 0.2,
    }

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
