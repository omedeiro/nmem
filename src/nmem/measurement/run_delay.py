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
    DEFAULT_SCOPE,
)

plt.rcParams["figure.figsize"] = [10, 12]


if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_delay"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)
    current_cell = measurement_settings["cell"]

    current_settings = CELLS[current_cell]

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 0,
        "read_width": 10,
        "enable_write_width": 4,
        "enable_read_width": 4,
        "enable_write_phase": -7,
        "enable_read_phase": -7,
        "bitmsg_channel": "N0NNRN1NNR",
        "bitmsg_enable": "NWNNENWNNE",
    }

    scope_settings = {
        "scope_horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "scope_timespan": HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS,
        "scope_num_samples": NUM_SAMPLES,
        "scope_sample_rate": NUM_SAMPLES / (HORIZONTAL_SCALE[FREQ_IDX] * NUM_DIVISIONS),
    }

    NUM_MEAS = 1000

    measurement_settings.update(
        {
            **waveform_settings,
            **current_settings,
            **scope_settings,
            "CELLS": CELLS,
            "HEATERS": HEATERS,
            "num_meas": NUM_MEAS,
            "spice_device_current": SPICE_DEVICE_CURRENT,
            "sweep_parameter_x": "enable_read_current",
            "sweep_parameter_y": "read_current",
        }
    )
    current_cell = measurement_settings.get("cell")

    nm.setup_scope_bert(
        b,
        measurement_settings,
        division_zero=(3.5, 4.3),
        division_one=(3.5, 4.3),
    )
    data_dict = nm.run_measurement_delay(
        b,
        measurement_settings,
        plot=True,
        delay=1,
    )


    file_path, time_str = qf.save(
        b.properties, data_dict.get("measurement_name"), data_dict
    )

    # if data_dict["sweep_x_len"] > 1 and data_dict["sweep_y_len"] > 1:
    #     fig, ax = plt.subplots()
    #     nm.plot_array(
    #         ax,
    #         data_dict,
    #         "bit_error_rate",
    #     )

    #     fig, ax = plt.subplots()
    #     nm.plot_array(
    #         ax,
    #         data_dict,
    #         "write_0_read_1_norm",
    #     )

    #     fig, ax = plt.subplots()
    #     nm.plot_array(
    #         ax,
    #         data_dict,
    #         "write_1_read_0_norm",
    #     )
    #     fig = nm.plot_header(fig, data_dict)
    #     fig.savefig(f"{file_path}_ber_sweep.png")
    # else:
    #     fig, ax = plt.subplots()
    #     nm.plot_slice(
    #         ax,
    #         data_dict,
    #         "bit_error_rate",
    #     )
    #     fig = nm.plot_header(fig, data_dict)
    #     fig.savefig(f"{file_path}_ber_sweep.png")

    nm.write_dict_to_file(file_path, measurement_settings)

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
    print(f"Bit Error Rate: {data_dict['bit_error_rate']}")
    nm.set_awg_off(b)
