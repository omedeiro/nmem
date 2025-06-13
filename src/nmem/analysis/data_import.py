import os

import ltspice
import numpy as np
import pandas as pd
import scipy.io as sio

from nmem.analysis.bit_error import (
    get_bit_error_rate,
    get_bit_error_rate_args,
)
from nmem.analysis.constants import IRM
from nmem.analysis.currents import (
    get_channel_temperature,
    get_enable_write_current,
    get_read_current,
    get_read_currents,
    get_write_current,
)
from nmem.analysis.plotting import CMAP
from nmem.analysis.utils import filter_first
from nmem.simulation.spice_circuits.functions import process_read_data


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
            y_die = 6 - (int(die_str[1]) - 1)  # '1' → 5, '7' → 0

            x_dev = ord(dev_str[0].upper()) - ord("A")  # 'A' → 0
            y_dev = 7 - (int(dev_str[1]) - 1)  # '1' → 7 → 0

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
            records.append(
                {
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
                }
            )

        except Exception as e:
            print(f"Skipping malformed entry: die={die}, dev={dev}, error={e}")

    return pd.DataFrame(records)


def import_read_current_sweep_operating_data(directory):
    dict_list = import_directory(directory)
    ic_list = [IRM]
    write_current_list = [0]
    ic_list2 = [IRM]
    write_current_list2 = [0]
    for data_dict in dict_list:
        write_current = get_write_current(data_dict)
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        read_currents = get_read_currents(data_dict)
        if not np.isnan(berargs[0]) and write_current < 100:
            ic_list.append(read_currents[berargs[0]])
            write_current_list.append(write_current)
        if not np.isnan(berargs[2]) and write_current > 100:
            ic_list.append(read_currents[berargs[3]])
            write_current_list.append(write_current)
        if not np.isnan(berargs[1]):
            ic_list2.append(read_currents[berargs[1]])
            write_current_list2.append(write_current)
        if not np.isnan(berargs[3]):
            ic_list2.append(read_currents[berargs[2]])
            write_current_list2.append(write_current)
    return dict_list, ic_list, write_current_list, ic_list2, write_current_list2


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


def import_elionix_log(log_path):
    """
    Import and parse the Elionix log file, returning parsed dataframes and arrays for analysis.
    Returns:
        df_z: DataFrame of z heights
        df_rot_valid: DataFrame of valid rotations
        dx_nm, dy_nm: np.arrays of alignment deltas
        delta_table: DataFrame of all inter-pass deltas
    """
    import re

    import pandas as pd

    with open(log_path, "r", encoding="utf-8") as f:
        log_text = f.read()
        lines = log_text.splitlines()
    # Patterns
    device_header_re = re.compile(r"\d+\s+(WSP_[A-Za-z0-9_]+)")
    z_try_re = re.compile(r"z try:\s+(\d+)")
    xy_try_re = re.compile(r"xy try:\s+(\d+)")
    z_height_re = re.compile(r"z:\s+([\d.]+)\s+\[mm\]")
    rotation_re = re.compile(r"rotation:\s+([\d.]+)\s+\[mrad\]")
    car_block_pattern = re.compile(
        r"^\s*\d+\s+(?P<car_file>\w+\.car).*?(?=^\s*\d+\s+\w+\.car|\Z)",
        re.DOTALL | re.MULTILINE,
    )
    # Extract z height, retry count, rotation
    data = []
    rotations = []
    current_device = None
    current_entry = {}
    for line in lines:
        line = line.strip()
        if match := device_header_re.match(line):
            if current_entry:
                data.append(current_entry)
            current_device = match.group(1)
            current_entry = {
                "device": current_device,
                "xy_try": None,
                "z_try": None,
                "z_height_mm": None,
            }
        elif match := xy_try_re.search(line):
            current_entry["xy_try"] = int(match.group(1))
        elif match := z_try_re.search(line):
            current_entry["z_try"] = int(match.group(1))
        elif match := z_height_re.search(line):
            current_entry["z_height_mm"] = float(match.group(1))
        elif match := rotation_re.search(line):
            if current_device:
                rotations.append(
                    {"device": current_device, "rotation_mrad": float(match.group(1))}
                )
    if current_entry:
        data.append(current_entry)
    df_z = pd.DataFrame(data).dropna(subset=["z_height_mm"])
    df_rot = pd.DataFrame(rotations)
    df_rot_valid = df_rot[df_rot["rotation_mrad"] >= 1.0]
    # Inter-pass alignment deltas
    inter_pass_deltas = []
    for match in car_block_pattern.finditer(log_text):
        car_file = match.group("car_file")
        block_text = match.group(0)
        search_matches = re.findall(
            r"alignment search (\d)-([A-Z])[\s\S]*?searched position:\s+([\d.]+)\s+([\d.]+)",
            block_text,
        )
        mark_dict = {}
        for pass_id, mark_id, x_str, y_str in search_matches:
            key = mark_id.upper()
            mark_dict.setdefault(key, {})[pass_id] = (float(x_str), float(y_str))
        for mark_id, passes in mark_dict.items():
            if "1" in passes and "2" in passes:
                (x1, y1), (x2, y2) = passes["1"], passes["2"]
                dx_nm = (x2 - x1) * 1e6
                dy_nm = (y2 - y1) * 1e6
                inter_pass_deltas.append(
                    {
                        "car_file": car_file,
                        "mark_id": mark_id,
                        "dx_nm": round(dx_nm, 2),
                        "dy_nm": round(dy_nm, 2),
                    }
                )
    delta_table = pd.DataFrame(inter_pass_deltas)
    dx_nm = delta_table["dx_nm"].to_numpy()
    dy_nm = delta_table["dy_nm"].to_numpy()
    return df_z, df_rot_valid, dx_nm, dy_nm, delta_table


def import_geom_loop_size_data(data_dir="data"):
    """
    Imports and processes loop size data for geometry analysis.
    Returns:
        data: list of dicts from import_directory
        loop_sizes: np.array of loop sizes
    """
    from nmem.analysis.data_import import import_directory

    loop_sizes = np.arange(1.7, 5.2, 0.5)
    data = import_directory(data_dir)
    return data, loop_sizes


def load_and_clean_thickness(path):
    df = pd.read_csv(path)
    df["d(nm)"] = (
        df["d(nm)"].astype(str).str.extract(r"([-+]?\d*\.\d+|\d+)").astype(float)
    )
    grouped = df.groupby(["Y", "X"])["d(nm)"].mean().reset_index()
    map_df = grouped.pivot(index="Y", columns="X", values="d(nm)")
    return map_df.drop(index="Y", errors="ignore").drop(columns="X", errors="ignore")


def import_read_current_sweep_data():
    """
    Import all relevant data lists for the read current sweep enable read analysis.
    Returns:
        dict_list: list of dicts for enable read (choose one for plotting)
        data_list: list of dicts for enable write
        data_list2: selected subset of data_list for plotting
    """
    data = import_directory("data")
    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    enable_read_310_C4_list = import_directory("data_310uA_C4")
    data_inverse = import_directory("data_inverse")
    dict_list = [enable_read_290_list, enable_read_300_list, enable_read_310_list]
    dict_list = dict_list[2]  # Use 310uA by default
    data_list = import_directory("../read_current_sweep_enable_write/data")
    data_list2 = [data_list[0], data_list[3], data_list[-6], data_list[-1]]
    return dict_list, data_list, data_list2


def import_read_current_sweep_three_data():
    """
    Import all relevant data lists for the three read current sweep analysis.
    Returns:
        dict_list: list of dicts for enable read (290uA, 300uA, 310uA)
    """
    enable_read_290_list = import_directory("data_290uA")
    enable_read_300_list = import_directory("data_300uA")
    enable_read_310_list = import_directory("data_310uA")
    return [enable_read_290_list, enable_read_300_list, enable_read_310_list]


def import_read_current_sweep_enable_write_data():
    """
    Import all relevant data lists for the read current sweep enable write analysis.
    Returns:
        data_list: list of dicts for all data
        data_list2: selected subset of data_list for plotting
        colors: color array for plotting
    """
    data_list = import_directory("data")
    data_list2 = [data_list[0], data_list[3], data_list[-6]]
    colors = CMAP(np.linspace(0, 1, 4))
    return data_list, data_list2, colors


def import_simulation_data(data_dir="data"):
    """Import and sort .raw simulation files by write current."""
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".raw")]
    write_current_list = []
    for file in files:
        data = ltspice.Ltspice(f"{data_dir}/{file}").parse()
        ltsp_data_dict = process_read_data(data)
        write_current = ltsp_data_dict[0]["write_current"][0]
        write_current_list.append(write_current * 1e6)
    sorted_args = np.argsort(write_current_list)
    sorted_files = [files[i] for i in sorted_args]
    return sorted_files


def import_read_current_sweep_sim_data(
    data_dir="data",
    write_current_dir="../read_current_sweep_write_current2/write_current_sweep_C3",
):
    # get raw files
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".raw")]
    # Sort files by write current
    write_current_list = []
    for file in files:
        data = ltspice.Ltspice(f"{data_dir}/{file}").parse()
        ltsp_data_dict = process_read_data(data)
        write_current = ltsp_data_dict[0]["write_current"][0]
        write_current_list.append(write_current * 1e6)
    sorted_args = np.argsort(write_current_list)
    files = [files[i] for i in sorted_args]
    ltsp_data = ltspice.Ltspice("nmem_cell_read_example_trace.raw").parse()
    ltsp_data_dict = process_read_data(ltsp_data)
    dict_list = import_directory(write_current_dir)
    write_current_list2 = []
    for data_dict in dict_list:
        write_current = filter_first(data_dict["write_current"])
        write_current_list2.append(write_current * 1e6)
    sorted_args2 = np.argsort(write_current_list2)
    dict_list = [dict_list[i] for i in sorted_args2]
    write_current_list2 = [write_current_list2[i] for i in sorted_args2]
    return files, ltsp_data_dict, dict_list, write_current_list2


def load_current_sweep_data():
    # get raw files
    files = os.listdir("data")
    files = [f for f in files if f.endswith(".raw")]
    # Sort files by write current
    write_current_list = []
    for file in files:
        data = ltspice.Ltspice(f"data/{file}").parse()
        ltsp_data_dict = process_read_data(data)
        write_current = ltsp_data_dict[0]["write_current"][0]
        write_current_list.append(write_current * 1e6)

    sorted_args = np.argsort(write_current_list)
    files = [files[i] for i in sorted_args]

    ltsp_data = ltspice.Ltspice("nmem_cell_read_example_trace.raw").parse()
    ltsp_data_dict = process_read_data(ltsp_data)

    dict_list = import_directory(
        "../read_current_sweep_write_current2/write_current_sweep_C3"
    )
    dict_list = dict_list[::2]
    write_current_list = []
    for data_dict in dict_list:
        write_current = filter_first(data_dict["write_current"])
        write_current_list.append(write_current * 1e6)

    sorted_args = np.argsort(write_current_list)
    dict_list = [dict_list[i] for i in sorted_args]
    write_current_list = [write_current_list[i] for i in sorted_args]

    return files, ltsp_data_dict, dict_list, write_current_list


def import_operating_data(directory):
    dict_list = import_directory(directory)
    ic_list = []
    write_current_list = []
    ic_list2 = []
    write_current_list2 = []
    for data_dict in dict_list:
        write_current = get_write_current(data_dict)
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        read_currents = get_read_currents(data_dict)
        if not np.isnan(berargs[0]) and write_current < 100:
            ic_list.append(read_currents[berargs[0]])
            write_current_list.append(write_current)
        if not np.isnan(berargs[2]) and write_current > 100:
            ic_list.append(read_currents[berargs[3]])
            write_current_list.append(write_current)
        if not np.isnan(berargs[1]):
            ic_list2.append(read_currents[berargs[1]])
            write_current_list2.append(write_current)
        if not np.isnan(berargs[3]):
            ic_list2.append(read_currents[berargs[2]])
            write_current_list2.append(write_current)
    return dict_list, ic_list, write_current_list, ic_list2, write_current_list2


def import_delay_data(data_dir="data3"):
    """Import and preprocess data for retention test plots."""
    dict_list = import_directory(data_dir)
    delay_list = []
    bit_error_rate_list = []
    for data_dict in dict_list:
        delay = data_dict.get("delay").flatten()[0] * 1e-3
        bit_error_rate = get_bit_error_rate(data_dict)
        delay_list.append(delay)
        bit_error_rate_list.append(bit_error_rate)
    fidelity = 1 - np.array(bit_error_rate_list)
    return np.array(delay_list), np.array(bit_error_rate_list), fidelity
