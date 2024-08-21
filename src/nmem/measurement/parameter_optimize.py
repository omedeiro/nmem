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
from nmem.measurement.cells import CELLS, HEATERS, SPICE_DEVICE_CURRENT

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

NUM_MEAS = 100
NUM_DIVISIONS = 10
NUM_SAMPLES = 5e3
NUM_POINTS = 2**8
ENABLE_WIDTH = 100e-9
BIAS_WIDTH = 300e-9


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
    data_dict = nm.run_measurement(b, meas_dict, plot=False)

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
        Real(10, 100, name="write_current"),
        Real(640, 710, name="read_current"),
        Real(180, 230, name="enable_write_current"),
        Real(140, 200, name="enable_read_current"),
    ]
    # space = [
    #     Integer(50, 200, name="write_width"),
    #     Integer(50, 150, name="read_width"),
    #     Integer(10, 50, name="enable_write_width"),
    #     Integer(10, 50, name="enable_read_width"),
    #     Integer(-50, 50, name="enable_write_phase"),
    #     Integer(-50, 50, name="enable_read_phase"),
    # ]
    nm.setup_scope_bert(b, meas_dict)
    opt_result = gp_minimize(
        partial(objective_bias, meas_dict=meas_dict),
        space,
        n_calls=100,
        verbose=True,
        x0=[
            45,
            688,
            223,
            171,
        ],
        # x0=[
        #     180,
        #     68,
        #     50,
        #     50,
        #     -50,
        #     -34,
        # ],
    )

    return opt_result, meas_dict


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
        "write_width": 180,
        "read_width": 68,  #
        "enable_write_width": 50,
        "enable_read_width": 50,
        "enable_write_phase": -50,
        "enable_read_phase": -34,
    }

    # waveform_settings = {
    #     "num_points": 256,
    #     "write_width": 90,
    #     "read_width": 90,  #
    #     "enable_write_width": 30,
    #     "enable_read_width": 30,
    #     "enable_write_phase": 0,
    #     "enable_read_phase": 30,
    # }

    measurement_settings = {
        **waveform_settings,
        "measurement_name": measurement_name,
        "sample_name": sample_name,
        "cell": current_cell,
        "CELLS": CELLS,
        "HEATERS": HEATERS,
        "num_meas": NUM_MEAS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "sample_time": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "num_samples_scope": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
        "spice_device_current": SPICE_DEVICE_CURRENT,
        "x": 0,
        "y": 0,
        "write_current": 45e-6,
        "read_current": 688e-6,
        "enable_write_current": 223e-6,
        "enable_read_current": 162e-6,
        "threshold_bert": 0.2,
        "bitmsg_channel": "N0RNR1RNRN",
        "bitmsg_enable": "NWNWEWNWEW",
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
