# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:17:34 2023

@author: omedeiro
"""

import time

import numpy as np
import qnnpy.functions.functions as qf
from matplotlib import pyplot as plt

import nmem.measurement.functions as nm
from nmem.calculations.calculations import calculate_critical_current
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


def read_sweep_scaled(
    measurement_settings, current_cell, num_points=15, start=0.8, end=0.95
):
    read_critical_current = (
        calculate_critical_current(
            measurement_settings["enable_read_current"] * 1e6, CELLS[current_cell]
        )
        * 1e-6
    )
    measurement_settings["y"] = np.linspace(
        read_critical_current * start, read_critical_current * end, num_points
    )
    return measurement_settings


def write_sweep_scaled(
    measurement_settings, current_cell, num_points=15, start=0.8, end=0.95
):
    write_critical_current = (
        calculate_critical_current(
            measurement_settings["enable_write_current"] * 1e6, CELLS[current_cell]
        )
        * 1e-6
    )
    measurement_settings["y"] = np.linspace(
        write_critical_current * start, write_critical_current * end, num_points
    )
    return measurement_settings


if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_parameter_sweep"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    current_cell = measurement_settings["cell"]


    slow_write = {
        "write_width": 40,
        "enable_write_width": 40,
        "enable_write_phase": 0,
    }
    fast_write = {
        "write_width": 0,
        "enable_write_width": 3,
        "enable_write_phase": -5,
    }

    slow_read = {
        "read_width": 40,
        "enable_read_width": 120,
        "enable_read_phase": 25,
    }

    fast_read = {
        "read_width": 7,
        "enable_read_width": 7,
        "enable_read_phase": -8,
    }

    two_nulls = {
        "bitmsg_channel": "N0NNRN1NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }
    two_nulls_inv = {
        "bitmsg_channel": "N1NNRN0NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }

    one_null = {
        "bitmsg_channel": "NN0NRNN1NR",
        "bitmsg_enable": "NNWNENNWNE",
    }

    zero_nulls = {
        "bitmsg_channel": "NNN0RNNN1R",
        "bitmsg_enable": "NNNWENNNWE",
    }

    two_emulate = {
        "bitmsg_channel": "N0RNRN1RNR",
        "bitmsg_enable": "NWNWENWNWE",
    }
    two_emulate_inv = {
        "bitmsg_channel": "N0NRRN1NRR",
        "bitmsg_enable": "NWWNENWWNE",
    }
    two_emulate_read = {
        "bitmsg_channel": "R0RRRR1RRR",
        "bitmsg_enable": "NWNNENWNNE",
    }
    two_emulate_ewrite = {
        "bitmsg_channel": "N0NNRN1NNR",
        "bitmsg_enable": "WWWWEWWWWE",
    }

    waveform_settings = {   
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        **fast_write,
        **fast_read,
        **two_emulate_read,
        "threshold_bert": 0.33,
        "threshold_enforced": 0.33,
    }

    current_settings = {
        "write_current": 40e-6,
        "read_current": 750e-6,
        "enable_write_current": 450e-6,
        "enable_read_current": 220e-6,
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }

    NUM_MEAS = 10000
    sweep_length = 31

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
        }
    )
    parameter_x = "enable_write_current"
    measurement_settings["x"] = np.array([measurement_settings[parameter_x]])
    # measurement_settings["x"] = np.linspace(100e-6, 450e-6, sweep_length)
    measurement_settings[parameter_x] = measurement_settings["x"][0]

    read_sweep = True
    if read_sweep:
        parameter_y = "read_current"
        # measurement_settings = read_sweep_scaled(
        #     measurement_settings, current_cell, sweep_length, start=0.7, end=1.10
        # )
        measurement_settings["y"] = np.array([current_settings["read_current"]])
        # measurement_settings["y"] = np.linspace(500e-6, 900e-6, sweep_length)

        measurement_settings[parameter_y] = measurement_settings["y"][0]
    else:
        parameter_y = "write_current"
        # measurement_settings = write_sweep_scaled(
        #     measurement_settings, current_cell, sweep_length, start=0.5, end=1.0
        # )
        # measurement_settings["y"] = np.array([current_settings["write_current"]])
        measurement_settings["y"] = np.linspace(00e-6, 60e-6, sweep_length)

        measurement_settings[parameter_y] = measurement_settings["y"][0]

    save_dict = nm.run_sweep(
        b,
        measurement_settings,
        parameter_x,
        parameter_y,
        plot_measurement=True,
        division_zero=(3.5, 5.0),
        division_one=(5.5, 7.0),
    )

    b.properties["measurement_settings"] = measurement_settings

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

    b.inst.awg.set_output(True, 1)
    b.inst.awg.set_output(True, 2)

    nm.write_dict_to_file(file_path, measurement_settings)

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
