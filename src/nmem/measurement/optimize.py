import numpy as np
import qnnpy.functions.functions as qf
from qnnpy.functions.ntron import nTron
from skopt.space import Integer, Real

import nmem.measurement.functions as nm
from nmem.measurement.cells import (
    CONFIG,
)


def update_space(meas_dict: dict, space: list, x0: list):
    for i, s in enumerate(space):
        if s.name in meas_dict:
            if meas_dict[s.name] < 1e-3:
                meas_dict[s.name] = x0[i] * 1e-6
            else:
                meas_dict[s.name] = x0[i]
    return meas_dict


def objective_primary(w1r0: np.ndarray, w0r1: np.ndarray, num_meas: float) -> float:
    errors = w1r0[0] + w0r1[0]
    res = errors / (num_meas * 2)
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
        Real(20, 120, name="write_current"),
        Real(600, 720, name="read_current"),
        Real(300, 420, name="enable_write_current"),
        Real(190, 280, name="enable_read_current"),
    ]

    x0 = [20, 685, 420, 200]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_enable(meas_dict: dict):
    measurement_name = "nMem_optimize_enable"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(550, 570, name="enable_write_current"),
        Real(280, 310, name="enable_read_current"),
    ]

    x0 = [560.0, 290.0]
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
        Real(600, 690, name="read_current"),
        Real(240, 320, name="read_enable_current"),
    ]

    x0 = [645, 290]
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
        Real(20, 40, name="write_current"),
        Real(540, 560, name="enable_write_current"),
    ]
    x0 = [33, 554]
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

    data_dict, measurement_settings = nm.run_measurement(b, meas_dict, plot=False)

    qf.save(b.properties, meas_dict["measurement_name"], data_dict)

    return objective_primary(
        data_dict["write_1_read_0"], data_dict["write_0_read_1"], meas_dict["num_meas"]
    )
