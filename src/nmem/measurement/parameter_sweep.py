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


if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_parameter_sweep"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    current_cell = measurement_settings["cell"]

    two_nulls = {
        "bitmsg_channel": "N0NNRN1NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }
    two_nulls_inv = {
        "bitmsg_channel": "N1NNRN0NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }
    two_nulls_zero = {
        "bitmsg_channel": "N0NNRN0NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }
    two_nulls_one = {
        "bitmsg_channel": "N1NNRN1NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }

    two_nulls_read = {
        "bitmsg_channel": "R0NNRR1NNR",
        "bitmsg_enable": "EWNNEEWNNE",
    }
    three_nulls = {
        "bitmsg_channel": "0NNNR1NNNR",
        "bitmsg_enable": "WNNNEWNNNE",
    }

    one_null = {
        "bitmsg_channel": "NN0NRNN1NR",
        "bitmsg_enable": "NNWNENNWNE",
    }

    zero_nulls = {
        "bitmsg_channel": "NNN0RNNN1R",
        "bitmsg_enable": "NNNWENNNWE",
    }
    zero_nulls_inv = {
        "bitmsg_channel": "NNN1RNNN0R",
        "bitmsg_enable": "NNNWENNNWE",
    }
    zero_nulls_zero = {
        "bitmsg_channel": "NNN0RNNN0R",
        "bitmsg_enable": "NNNWENNNWE",
    }
    zero_nulls_one = {
        "bitmsg_channel": "NNN1RNNN1R",
        "bitmsg_enable": "NNNWENNNWE",
    }
    zero_nulls_ewrite = {
        "bitmsg_channel": "NNN0RNNN1R",
        "bitmsg_enable": "WWWWEWWWWE",
    }
    zero_nulls_emulate = {
        "bitmsg_channel": "RRR0RRRR1R",
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
    two_emulate_eread = {
        "bitmsg_channel": "N0NNRN1NNR",
        "bitmsg_enable": "EWEEEEWEEE",
    }
    two_read = {
        "bitmsg_channel": "N1R0RN0R1R",
        "bitmsg_enable": "NWEWENWEWE",
    }
    two_read_inv = {
        "bitmsg_channel": "N0R0RN1R1R",
        "bitmsg_enable": "NWEWENWEWE",
    }

    slow_write = {
        "write_width": 40,
        "enable_write_width": 4,
        "enable_write_phase": -7,
    }
    fast_write = {
        "write_width": 0,
        "enable_write_width": 4,
        "enable_write_phase": -7,
    }
    slow_read = {
        "read_width": 40,
        "enable_read_width": 4,
        "enable_read_phase": -7,
    }
    fast_read = {
        "read_width": 10,
        "enable_read_width": 4,
        "enable_read_phase": -7,
    }

    write_byte = {
        "bitmsg_channel": "NNNN0R1R0R0R1R1R1R1RNNNNNNNN0R1R0R0R1R1R0R1RNNNN",
        "bitmsg_enable": "NNNNWEWEWEWEWEWEWEWENNNNNNNNWEWEWEWEWEWEWEWENNNN",
    }

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        **fast_write,
        **fast_read,
        **two_nulls,
        "voltage_threshold": 0.35,
    }

    current_settings = {
        "write_current": 155e-6,
        "read_current": 750e-6,
        "enable_write_current": 410e-6,
        "enable_read_current": 180e-6,
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }

    NUM_MEAS = 500
    sweep_length = 3

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
    # measurement_settings["x"] = np.linspace(00e-6, 550e-6, sweep_length)
    measurement_settings[parameter_x] = measurement_settings["x"][0]

    read_sweep = True
    if read_sweep:
        parameter_y = "read_current"
        # measurement_settings["y"] = np.array([current_settings["read_current"]])
        measurement_settings["y"] = np.linspace(750e-6, 850e-6, sweep_length)
        measurement_settings[parameter_y] = measurement_settings["y"][0]
    else:
        parameter_y = "write_current"
        # measurement_settings["y"] = np.array([current_settings["write_current"]])
        measurement_settings["y"] = np.linspace(100e-6, 200e-6, sweep_length)
        measurement_settings[parameter_y] = measurement_settings["y"][0]

    data_dict = nm.run_sweep(
        b,
        measurement_settings,
        parameter_x,
        parameter_y,
        plot_measurement=True,
        division_zero=(5.9, 6.5),
        division_one=(5.9, 6.5),
    )

    b.properties["measurement_settings"] = measurement_settings

    file_path, time_str = qf.save(
        b.properties, measurement_settings["measurement_name"], data_dict
    )

    fig, ax = plt.subplots()
    nm.plot_ber_sweep(
        ax,
        data_dict,
        parameter_x,
        parameter_y,
        "bit_error_rate",
    )
    nm.plot_ber_sweep(
        ax,
        data_dict,
        parameter_x,
        parameter_y,
        "write_0_read_1_norm",
    )
    nm.plot_ber_sweep(
        ax,
        data_dict,
        parameter_x,
        parameter_y,
        "write_1_read_0_norm",
    )
    nm.plot_header(fig, data_dict)
    fig.subplots_adjust(top=0.85)

    nm.set_awg_off(b)
    nm.write_dict_to_file(file_path, measurement_settings)

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
