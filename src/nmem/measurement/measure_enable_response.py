import time

import numpy as np
import qnnpy.functions.functions as qf
import qnnpy.functions.ntron as nt
from matplotlib import pyplot as plt

import nmem.measurement.functions as nm
from nmem.measurement.parameter_sweep import CONFIG, HORIZONTAL_SCALE, SAMPLE_RATE

plt.rcParams["figure.figsize"] = [10, 12]


if __name__ == "__main__":
    b = nt.nTron(CONFIG)
    REAL_TIME = 1
    NUM_MEAS = 100
    FREQ_IDX = 1

    sample_name = [
        b.sample_name,
        b.device_type,
        b.device_name,
        b.properties["Save File"]["cell"],
    ]
    sample_name = str("-".join(sample_name))
    date_str = time.strftime("%Y%m%d")
    measurement_name = f"{date_str}_measure_enable_response"

    measurement_settings = {
        "measurement_name": measurement_name,
        "sample_name": sample_name,
        "write_current": 205e-6,
        "read_current": 590e-6,
        "enable_voltage": 0.0,
        "enable_write_current": 132e-6,
        "enable_read_current": 150e-6,
        "num_meas": NUM_MEAS,
        "threshold_read": 100e-3,
        "threshold_enab": 15e-3,
        "threshold_bert": 0.15,
        "sample_rate": SAMPLE_RATE[FREQ_IDX],
        "horizontal_scale": HORIZONTAL_SCALE[FREQ_IDX],
        "sample_time": HORIZONTAL_SCALE[FREQ_IDX] * 10,  # 10 divisions
        "num_samples_scope": 5e3,
        "scope_sample_rate": 5e3 / (HORIZONTAL_SCALE[FREQ_IDX] * 10),
        "realtime": REAL_TIME,
        "x": 0,
        "y": 0,
        "num_samples": 2**8,
        "write_width": 100,
        "read_width": 100,  #
        "enable_write_width": 30,
        "enable_read_width": 30,
        "enable_write_phase": 0,
        "enable_read_phase": 30,
        "bitmsg_channel": "N01NRNNNRN",
        "bitmsg_enable": "NNNNENNNEE",
    }

    t1 = time.time()
    parameter_x = "enable_read_current"
    # measurement_settings["x"] = np.array([10e-6])
    measurement_settings["x"] = np.linspace(200e-6, 400e-6, 5)
    parameter_y = "read_current"
    # measurement_settings["y"] = [315e-6]
    measurement_settings["y"] = np.linspace(200e-6, 700e-6, 21)

    b, measurement_settings, save_dict = nm.run_sweep(
        b, measurement_settings, parameter_x, parameter_y, plot_measurement=True
    )
    file_path, time_str = qf.save(b.properties, measurement_name, save_dict)
    save_dict["time_str"] = time_str
    nm.plot_ber_sweep(
        save_dict,
        measurement_settings,
        file_path,
        parameter_x,
        parameter_y,
        "ber",
    )

    t2 = time.time()
    print(f"run time {(t2-t1)/60:.2f} minutes")
    b.inst.scope.save_screenshot(
        f"{file_path}_scope_screenshot.png", white_background=False
    )
    b.inst.awg.set_output(False, 1)
    b.inst.awg.set_output(False, 2)

    nm.write_dict_to_file(file_path, save_dict)
