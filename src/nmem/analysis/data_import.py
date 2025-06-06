import os

import numpy as np
import pandas as pd
import scipy.io as sio

from nmem.analysis.currents import (
    get_channel_temperature,
    get_enable_write_current,
    get_read_current,
    get_read_currents,
    get_write_current,
)

from nmem.analysis.bit_error import (
    get_bit_error_rate,
    get_bit_error_rate_args,
)


def load_autoprobe_data(filepath, grid_size=56):
    """Load autoprobe data from a parsed .mat file and return as a DataFrame with bounds-checked coordinates."""
    mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    die_name = mat["die_name"].flatten()
    device_name = mat["device_name"].flatten()
    data = mat["data"]

    Rmean = data.Rmean.flatten()
    Rmse = data.Rmse.flatten()

    records = []

    for die, dev, rmean, rmse in zip(die_name, device_name, Rmean, Rmse):
        try:
            die_str = str(die)
            dev_str = str(dev)

            # Parse and flip die/device coordinates
            x_die = ord(die_str[0].upper()) - ord("A")  # 'A' → 0
            y_die = 6 - (int(die_str[1]) - 1)           # '1' → 5, '7' → 0

            x_dev = ord(dev_str[0].upper()) - ord("A")  # 'A' → 0
            y_dev = 7 - (int(dev_str[1]) - 1)           # '1' → 7 → 0

            x_abs = x_die * 8 + x_dev
            y_abs = y_die * 8 + y_dev

            # Bounds and value checks
            if not (0 <= x_abs < grid_size and 0 <= y_abs < grid_size):
                raise ValueError(f"Out of bounds: ({x_abs}, {y_abs})")

            if not np.isfinite(rmean) or rmean < 0:
                continue  # skip bad resistance values
            if rmse > 0.5:
                rmean = np.nan  # set high Rmse to NaN

            if y_die == 5:
                squares = 50 * (x_dev + 1)
            else:
                squares = None
            records.append({
                "id": f"{die_str}_{dev_str}",
                "die": die_str,
                "device": dev_str,
                "x_die": x_die,
                "y_die": y_die,
                "x_dev": x_dev,
                "y_dev": y_dev,
                "x_abs": x_abs,
                "y_abs": y_abs,
                "Rmean": rmean,
                "Rmse": rmse,
                "squares": squares,
            })

        except Exception as e:
            print(f"Skipping malformed entry: die={die}, dev={dev}, error={e}")

    return pd.DataFrame(records)




def import_write_sweep_formatted() -> list[dict]:
    dict_list = import_directory(
        os.path.join(os.path.dirname(__file__), "write_current_sweep_enable_write/data")
    )
    dict_list = dict_list[1:]
    dict_list = dict_list[::-1]
    dict_list = sorted(
        dict_list, key=lambda x: x.get("enable_write_current").flatten()[0]
    )
    return dict_list


def import_delay_dict() -> dict:
    dict_list = import_directory(
        os.path.join(os.path.dirname(__file__), "read_delay_v2/data3")
    )
    delay_list = []
    bit_error_rate_list = []
    for data_dict in dict_list:
        delay = data_dict.get("delay").flatten()[0] * 1e-3
        bit_error_rate = get_bit_error_rate(data_dict)

        delay_list.append(delay)
        bit_error_rate_list.append(bit_error_rate)

    delay_dict = {}
    delay_dict["delay"] = delay_list
    delay_dict["bit_error_rate"] = bit_error_rate_list
    return delay_dict


def import_write_sweep_formatted_markers(dict_list) -> list[dict]:
    data = []
    data2 = []
    for data_dict in dict_list:
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_currents = get_read_currents(
            data_dict
        )  # This is correct. "y" is the write current in this .mat.
        enable_write_current = get_enable_write_current(data_dict)
        read_current = get_read_current(data_dict)
        write_current = get_write_current(data_dict)

        for i, arg in enumerate(berargs):
            if arg is not np.nan:

                if i == 0:
                    data.append(
                        {
                            "write_current": write_currents[arg],
                            "write_temp": get_channel_temperature(data_dict, "write"),
                            "read_current": read_current,
                            "enable_write_current": enable_write_current,
                        }
                    )
                if i == 2:
                    data2.append(
                        {
                            "write_current": write_currents[arg],
                            "write_temp": get_channel_temperature(data_dict, "write"),
                            "read_current": read_current,
                            "enable_write_current": enable_write_current,
                        }
                    )
    data_dict = {
        "data": data,
        "data2": data2,
    }
    return data_dict


def import_directory(file_path: str) -> list[dict]:
    dict_list = []
    files = get_file_names(file_path)
    files = sorted(files)
    for file in files:
        data = sio.loadmat(os.path.join(file_path, file))
        dict_list.append(data)

    save_directory_list(file_path, files)
    return dict_list


def save_directory_list(file_path: str, file_list: list[str]) -> None:
    with open(os.path.join(file_path, "data.txt"), "w") as f:
        for file_name in file_list:
            f.write(file_name + "\n")

    f.close()

    return



def get_file_names(file_path: str) -> list:
    files = os.listdir(file_path)
    files = [file for file in files if file.endswith(".mat")]
    return files
