import time

import numpy as np
import qnnpy.functions.functions as qf
from matplotlib import pyplot as plt

import nmem.measurement.functions as nm
from nmem.analysis.enable_current_relation import find_peak
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
from nmem.measurement.parameter_sweep import CONFIG, construct_currents

plt.rcParams["figure.figsize"] = [10, 12]



if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_measure_enable_response"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 40,
        "read_width": 40,
        "enable_write_width": 40,
        "enable_read_width": 120,
        "enable_write_phase": 0,
        "enable_read_phase": 0,
        "bitmsg_channel": "N0NNRNNNNR",
        "bitmsg_enable": "NNNNENNNNE",
    }

    current_settings = {
        "write_current": 0e-6,
        "read_current": 600e-6,
        "enable_write_current": 0e-6,
        "enable_read_current": 150e-6,
    }
    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }
    NUM_MEAS = 100

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
    current_cell = measurement_settings["cell"]

    parameter_x = "enable_read_current"
    # measurement_settings["x"] = np.array([250e-6])
    measurement_settings["x"] = np.linspace(200e-6, 300e-6, 8)
    parameter_y = "read_current"
    # measurement_settings["y"] = [400e-6]
    measurement_settings["y"] = np.linspace(300e-6, 600e-6, 61)
    print(f"Slope: {CELLS[current_cell]['slope']}")
    print(f"Intercept: {CELLS[current_cell]['intercept']}")
    measurement_settings["x_subset"] = measurement_settings["x"]
    measurement_settings["y_subset"] = construct_currents(
        measurement_settings["x"],
        CELLS[current_cell]["slope"],
        CELLS[current_cell]["intercept"] * 1e-6,
        0.1,
        CELLS[current_cell]["max_critical_current"],
    )

    save_dict = nm.run_sweep_subset(
        b, measurement_settings, parameter_x, parameter_y, plot_measurement=False
    )
    save_dict["trace_chan_in"] = save_dict["trace_chan_in"][:, :, 1]
    save_dict["trace_chan_out"] = save_dict["trace_chan_out"][:, :, 1]
    save_dict["trace_enab"] = save_dict["trace_enab"][:, :, 1]
    file_path, time_str = qf.save(
        b.properties, measurement_settings["measurement_name"], save_dict
    )
    save_dict["time_str"] = time_str
    nm.plot_ber_sweep(
        save_dict,
        measurement_settings,
        file_path,
        parameter_x,
        parameter_y,
        "bit_error_rate",
    )

    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    nm.write_dict_to_file(file_path, save_dict)
    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")

    find_peak(save_dict)
