from typing import List
import numpy as np
from skopt.space import Real, Integer
import nmem.measurement.functions as nm
import qnnpy.functions.functions as qf
from qnnpy.functions.ntron import nTron

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
        Real(50, 70, name="write_current"),
        Real(620, 640, name="read_current"),
        Real(330, 340, name="enable_write_current"),
        Real(230, 240, name="enable_read_current"),
    ]

    x0 = [60, 630, 335, 236]
    meas_dict = update_space(meas_dict, space, x0)
    return meas_dict, space, x0, b


def optimize_enable(meas_dict: dict):
    measurement_name = "nMem_optimize_enable"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    meas_dict.update(measurement_settings)
    space = [
        Real(150, 350, name="enable_write_current"),
        Real(150, 250, name="enable_read_current"),
    ]

    x0 = [325.0, 240.0]
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

    data_dict, measurement_settings = nm.run_measurement(b, meas_dict, plot=False)

    qf.save(b.properties, meas_dict["measurement_name"], data_dict)

    return objective_primary(
        data_dict["write_1_read_0"], data_dict["write_0_read_1"], meas_dict["num_meas"]
    )
