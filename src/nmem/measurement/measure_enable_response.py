import time

import numpy as np
import qnnpy.functions.functions as qf
import qnnpy.functions.ntron as nt
from matplotlib import pyplot as plt

import nmem.measurement.functions as nm
from nmem.analysis.enable_current_relation import find_peak
from nmem.calculations.calculations import htron_critical_current
from nmem.measurement.cells import CELLS
from nmem.measurement.parameter_sweep import CONFIG, HORIZONTAL_SCALE, SAMPLE_RATE

plt.rcParams["figure.figsize"] = [10, 12]


def construct_currents(
    heater_currents: np.ndarray,
    slope: float,
    intercept: float,
    margin: float,
    num_points: int = 4,
):
    bias_current_array = np.zeros((len(heater_currents), num_points))
    for heater_current in heater_currents:
        critical_current = htron_critical_current(heater_current, slope, intercept)
        bias_currents = np.linspace(
            critical_current * (1 - margin), critical_current * (1 + margin), num_points
        )
        bias_current_array[heater_currents == heater_current] = bias_currents

    return bias_current_array


if __name__ == "__main__":
    b = nt.nTron(CONFIG)
    REAL_TIME = 1
    NUM_MEAS = 100
    FREQ_IDX = 1
    current_cell = b.properties["Save File"]["cell"]
    sample_name = [
        b.sample_name,
        b.device_type,
        b.device_name,
        current_cell,
    ]
    sample_name = str("-".join(sample_name))
    date_str = time.strftime("%Y%m%d")
    measurement_name = f"{date_str}_measure_enable_response"

    measurement_settings = {
        "measurement_name": measurement_name,
        "sample_name": sample_name,
        "cell": current_cell,
        "write_current": 0e-6,
        "read_current": 600e-6,
        "enable_write_current": 0e-6,
        "enable_read_current": 150e-6,
        "num_meas": NUM_MEAS,
        "threshold_read": 100e-3,
        "threshold_enab": 15e-3,
        "threshold_bert": 0.2,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "sample_time": HORIZONTAL_SCALE[FREQ_IDX] * 10,  # 10 divisions
        "num_samples_scope": int(5000),
        "scope_sample_rate": int(5000) / (HORIZONTAL_SCALE[FREQ_IDX] * 10),
        "x": 0,
        "y": 0,
        "num_samples": 2**8,
        "write_width": 100,
        "read_width": 100,  #
        "enable_write_width": 30,
        "enable_read_width": 30,
        "enable_write_phase": 0,
        "enable_read_phase": 30,
        "bitmsg_channel": "NNNNRNNNRN",
        "bitmsg_enable": "NENNENNNEE",
    }

    t1 = time.time()
    parameter_x = "enable_read_current"
    # measurement_settings["x"] = np.array([280e-6])
    measurement_settings["x"] = np.linspace(200e-6, 300e-6, 9)
    parameter_y = "read_current"
    # measurement_settings["y"] = [400e-6]
    measurement_settings["y"] = np.linspace(300e-6, 900e-6, 61)

    measurement_settings["x_subset"] = measurement_settings["x"]
    measurement_settings["y_subset"] = construct_currents(
        measurement_settings["x"],
        CELLS[current_cell]["slope"],
        CELLS[current_cell]["intercept"] * 1e-6,
        0.12,
    )

    save_dict = nm.run_sweep_subset(
        b, measurement_settings, parameter_x, parameter_y, plot_measurement=False
    )
    save_dict["trace_chan_in"] = save_dict["trace_chan_in"][:, :, 1]
    save_dict["trace_chan_out"] = save_dict["trace_chan_out"][:, :, 1]
    save_dict["trace_enab"] = save_dict["trace_enab"][:, :, 1]
    file_path, time_str = qf.save(b.properties, measurement_name, save_dict)
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
