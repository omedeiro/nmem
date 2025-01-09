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
from nmem.measurement.functions import (
    construct_currents,
    build_array,
    get_fitting_points,
    plot_fitting,
    filter_plateau,
)

plt.rcParams["figure.figsize"] = [10, 12]


if __name__ == "__main__":
    t1 = time.time()
    measurement_name = "nMem_measure_enable_response"
    measurement_settings, b = nm.initilize_measurement(CONFIG, measurement_name)

    waveform_settings = {
        "num_points": NUM_POINTS,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "write_width": 40,
        "read_width": 7,
        "enable_write_width": 40,
        "enable_read_width": 4,
        "enable_write_phase": 0,
        "enable_read_phase": -7,
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
            "sweep_parameter_x": "enable_read_current",
            "sweep_parameter_y": "read_current",
            "voltage_threshold": 0.42,
        }
    )
    current_cell = measurement_settings.get("cell")

    # measurement_settings["x"] = np.linspace(200e-6, 300e-6, 15)
    # measurement_settings["y"] = np.linspace(00e-6, 1000e-6, 81)

    measurement_settings["x"] = np.linspace(0e-6, 500e-6, 21)
    measurement_settings["y"] = np.linspace(0e-6, 1000e-6, 81)

    measurement_settings["x_subset"] = measurement_settings.get("x")
    measurement_settings["y_subset"] = construct_currents(
        measurement_settings["x"],
        CELLS[current_cell]["slope"],
        CELLS[current_cell]["y-intercept"] * 1e-6,
        0.2,
        CELLS[current_cell]["max_critical_current"],
    )

    data_dict = nm.run_sweep_subset(
        b,
        measurement_settings,
        plot_measurement=False,
        division_zero=(5.9, 6.5),
        division_one=(5.9, 6.5),
    )
    file_path, time_str = qf.save(
        b.properties, data_dict.get("measurement_name"), data_dict
    )

    fig, ax = plt.subplots()
    nm.plot_array(
        ax,
        data_dict,
        "total_switches_norm",
    )
    cbar = fig.colorbar(ax.get_children()[0], ax=ax)
    cbar.set_label("Total Switches Normalized")
    nm.plot_header(fig, data_dict)
    plt.savefig(f"{file_path}_fit.png")
    plt.show()

    fig, ax = plt.subplots()
    nm.plot_slice(
        ax,
        data_dict,
        "total_switches_norm",
    )
    nm.plot_header(fig, data_dict)
    plt.savefig(f"{file_path}_slice.png")

    nm.set_awg_off(b)

    nm.write_dict_to_file(file_path, data_dict)
    print(f"run time {(time.time()-t1)/60:.2f} minutes")

    fig, axs = plt.subplots()
    x, y, ztotal = build_array(data_dict, "total_switches_norm")
    xfit, yfit = get_fitting_points(x, y, ztotal)
    xfit, yfit = filter_plateau(xfit, yfit, yfit[0]*0.9)
    plot_fitting(axs, xfit, yfit)
    axs.set_xlabel("Enable Current ($\mu$A)")
    axs.set_ylabel("Channel Current ($\mu$A)")
    axs.set_aspect("equal")
    nm.plot_header(fig, data_dict)
    plt.savefig(f"{file_path}_fitting.png")
