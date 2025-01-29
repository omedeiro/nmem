import collections
import os
from typing import List, Literal, Tuple

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from nmem.calculations.calculations import (
    calculate_heater_power,
    calculate_read_currents,
    htron_critical_current,
)
from nmem.measurement.cells import CELLS

SUBSTRATE_TEMP = 1.3
CRITICAL_TEMP = 12.3
CMAP = plt.get_cmap("plasma").reversed()


def build_array(
    data_dict: dict, parameter_z: Literal["total_switches_norm"]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data_dict.get("total_switches_norm") is None:
        data_dict["total_switches_norm"] = get_total_switches_norm(data_dict)
    x: np.ndarray = data_dict.get("x")[0][:, 0] * 1e6
    y: np.ndarray = data_dict.get("y")[0][:, 0] * 1e6
    z: np.ndarray = data_dict.get(parameter_z)

    xlength: int = filter_first(data_dict.get("sweep_x_len", len(x)))
    ylength: int = filter_first(data_dict.get("sweep_y_len", len(y)))

    # X, Y reversed in reshape
    zarray = z.reshape((ylength, xlength), order="F")
    return x, y, zarray


def filter_plateau(
    xfit: np.ndarray, yfit: np.ndarray, plateau_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    xfit = np.where(yfit < plateau_height, xfit, np.nan)
    yfit = np.where(yfit < plateau_height, yfit, np.nan)

    # Remove nans
    xfit = xfit[~np.isnan(xfit)]
    yfit = yfit[~np.isnan(yfit)]

    return xfit, yfit


def filter_first(value) -> any:
    if isinstance(value, collections.abc.Iterable) and not isinstance(
        value, (str, bytes)
    ):
        return np.asarray(value).flatten()[0]
    return value


def convert_cell_to_coordinates(cell: str) -> tuple:
    """Converts a cell name like 'A1' to coordinates (x, y)."""
    column_letter = cell[0]
    row_number = int(cell[1:]) - 1
    column_number = ord(column_letter) - ord("A")
    return column_number, row_number


def calculate_bit_error_rate(data_dict: dict) -> np.ndarray:
    num_meas = data_dict.get("num_meas")[0][0]
    w1r0 = data_dict.get("write_1_read_0")[0].flatten() / num_meas
    w0r1 = data_dict.get("write_0_read_1")[0].flatten() / num_meas
    ber = (w1r0 + w0r1) / 2
    return ber


def calculate_channel_temperature(data_dict: dict) -> np.ndarray:

    critical_temperature: float = data_dict.get("critical_temperature", CRITICAL_TEMP)
    substrate_temperature: float = data_dict.get(
        "substrate_temperature", SUBSTRATE_TEMP
    )

    ih = data_dict.get("x").flatten()[0] * 1e6
    cell = data_dict.get("cell")[0]
    ih_max = CELLS[cell]["x_intercept"]

    return _calculate_channel_temperature(
        critical_temperature, substrate_temperature, ih, ih_max
    )


def _calculate_channel_temperature(
    critical_temperature: float,
    substrate_temperature: float,
    ih: float,
    ih_max: float,
) -> np.ndarray:
    N = 2.0

    if ih_max == 0:
        raise ValueError("ih_max cannot be zero to avoid division by zero.")

    channel_temperature = (critical_temperature**4 - substrate_temperature**4) * (
        (ih / ih_max) ** N
    ) + substrate_temperature**4

    channel_temperature = np.maximum(channel_temperature, 0)

    temp_channel = np.power(channel_temperature, 0.25)
    return temp_channel


def calculate_critical_current(data_dict: dict) -> np.ndarray:
    critical_temperature = data_dict.get("critical_temperature", 12.3)
    channel_temperature = calculate_channel_temperature(data_dict)
    ic_zero = get_critical_current_zero_temp(data_dict)

    return _calculate_critical_current(
        critical_temperature, channel_temperature, ic_zero
    )


def _calculate_critical_current(
    critical_temperature: float,
    substrate_temperature: float,
    ic_max: float,
) -> np.ndarray:
    ic = ic_max * np.power(
        (1 - np.power(substrate_temperature / critical_temperature, 3)), 2.1
    )
    return ic


def create_trace_hist_plot(
    ax_dict: dict[str, Axes], dict_list: list[dict], save: bool = False
) -> Axes:
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()

    plot_trace_averaged(
        ax_dict["A"], dict_list[4], "trace_write_avg", color="#293689", label="Write"
    )
    plot_trace_averaged(
        ax2, dict_list[4], "trace_ewrite_avg", color="#ff1423", label="Enable Write"
    )
    plot_trace_averaged(
        ax_dict["B"], dict_list[4], "trace_read0_avg", color="#1966ff", label="Read 0"
    )
    plot_trace_averaged(
        ax_dict["B"],
        dict_list[4],
        "trace_read1_avg",
        color="#ff14f0",
        linestyle="--",
        label="Read 1",
    )
    plot_trace_averaged(
        ax3, dict_list[4], "trace_eread_avg", color="#ff1423", label="Enable Read"
    )

    plot_hist(ax_dict["C"], dict_list[3])

    ax_dict["A"].legend(loc="upper left")
    ax_dict["A"].set_ylabel("[mV]")
    ax2.legend()
    ax2.set_ylabel("[mV]")
    ax3.legend()
    ax3.set_ylabel("[mV]")
    ax_dict["B"].set_xlabel("Time [ns]")
    ax_dict["B"].set_ylabel("[mV]")
    ax_dict["B"].legend(loc="upper left")

    return ax_dict


def get_current_cell(data_dict: dict) -> str:
    cell = filter_first(data_dict.get("cell"))
    if cell is None:
        cell = filter_first(data_dict.get("sample_name"))[-2:]
    return cell


def get_critical_current_zero_temp(data_dict: dict) -> np.ndarray:
    cell = get_current_cell(data_dict)
    switching_current = CELLS[cell]["max_critical_current"] * 1e6
    substrate_temp = SUBSTRATE_TEMP
    critical_temp = CRITICAL_TEMP

    max_critical_current = switching_current / np.power(
        1 - np.power(substrate_temp / critical_temp, 3), 2.1
    )
    return max_critical_current


def get_enable_read_current(data_dict: dict) -> float:
    return filter_first(data_dict.get("enable_read_current")) * 1e6


def get_enable_write_current(data_dict: dict) -> float:
    return filter_first(data_dict.get("enable_write_current")) * 1e6


def get_enable_write_currents(data_dict: dict) -> np.ndarray:
    enable_write_currents = data_dict.get("x")[:, :, 0].flatten() * 1e6
    if len(enable_write_currents) == 1:
        enable_write_currents = data_dict.get("x")[:, 0].flatten() * 1e6

    if enable_write_currents[0] == enable_write_currents[1]:
        enable_write_currents = data_dict.get("y")[:, :, 0].flatten() * 1e6
    return enable_write_currents


def get_read_width(data_dict: dict) -> float:
    return filter_first(data_dict.get("read_width"))


def get_write_width(data_dict: dict) -> float:
    return filter_first(data_dict.get("write_width"))


def get_average_response(cell_dict):
    slope_list = []
    intercept_list = []
    for value in cell_dict.values():
        slope_list.append(value.get("slope"))
        intercept_list.append(value.get("y_intercept"))

    slope = np.mean(slope_list)
    intercept = np.mean(intercept_list)
    return slope, intercept


def get_file_names(file_path: str) -> list:
    files = os.listdir(file_path)
    files = [file for file in files if file.endswith(".mat")]
    return files


def get_text_from_bit(bit: str) -> str:
    if bit == "0":
        return "WR0"
    elif bit == "1":
        return "WR1"
    elif bit == "N":
        return ""
    elif bit == "R":
        return "RD"
    elif bit == "E":
        return "Read \nEnable"
    elif bit == "W":
        return "Write \nEnable"
    else:
        return None


def get_text_from_bit_v2(bit: str) -> str:
    if bit == "0":
        return "WR0"
    elif bit == "1":
        return "WR1"
    elif bit == "N":
        return ""
    elif bit == "R":
        return "RD"
    elif bit == "E":
        return "ER"
    elif bit == "W":
        return "EW"
    elif bit == "z":
        return "RD0"
    elif bit == "Z":
        return "W0R1"
    elif bit == "o":
        return "RD1"
    elif bit == "O":
        return "W1R0"
    else:
        return None


def get_text_from_bit_v3(bit: str) -> str:
    if bit == "0":
        return "0"
    elif bit == "1":
        return "1"
    elif bit == "N":
        return ""
    elif bit == "R":
        return ""
    elif bit == "E":
        return ""
    elif bit == "W":
        return ""
    else:
        return None


def get_total_switches_norm(data_dict: dict) -> np.ndarray:
    num_meas = data_dict.get("num_meas")[0][0]
    w0r1 = data_dict.get("write_0_read_1").flatten()
    w1r0 = num_meas - data_dict.get("write_1_read_0").flatten()
    total_switches_norm = w0r1 + w1r0
    return total_switches_norm


def get_read_currents(data_dict: dict) -> np.ndarray:
    read_currents = data_dict.get("y")[:, :, 0] * 1e6
    return read_currents.flatten()


def get_bit_error_rate(data_dict: dict) -> np.ndarray:
    return data_dict.get("bit_error_rate").flatten()


def get_critical_currents(dict_list: list) -> Tuple[list, list]:
    critical_currents = []
    critical_currents_std = []
    for data in dict_list:
        time = data.get("trace")[0, :]
        voltage = data.get("trace")[1, :]

        M = int(np.round(len(voltage), -2))
        if len(voltage) > M:
            voltage = voltage[:M]
            time = time[:M]
        else:
            voltage = np.concatenate([voltage, np.zeros(M - len(voltage))])
            time = np.concatenate([time, np.zeros(M - len(time))])

        current_time_trend = (
            data["vpp"]
            / 2
            / 10e3
            * (data["time_trend"][1, :])
            / (1 / (data["freq"] * 4))
            * 1e6
        )

        avg_critical_current = np.mean(current_time_trend)
        std_critical_current = np.std(current_time_trend)
        critical_currents.append(avg_critical_current)
        critical_currents_std.append(std_critical_current)

    return critical_currents, critical_currents_std


def get_fitting_points(
    x: np.ndarray, y: np.ndarray, ztotal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mid_idx = np.where(ztotal > np.nanmax(ztotal, axis=0) / 2)
    xfit, xfit_idx = np.unique(x[mid_idx[1]], return_index=True)
    yfit = y[mid_idx[0]][xfit_idx]
    return xfit, yfit


def get_max_enable_current(data_dict: dict) -> float:
    cell = get_current_cell(data_dict)
    return CELLS[cell]["x_intercept"]


def get_write_temperatures(data_dict: dict) -> np.ndarray:
    enable_write_currents = get_enable_write_currents(data_dict)
    max_enable_current = get_max_enable_current(data_dict)
    enable_write_temps = _calculate_channel_temperature(
        CRITICAL_TEMP, SUBSTRATE_TEMP, enable_write_currents, max_enable_current
    )
    return enable_write_temps


def get_write_temperature(data_dict: dict) -> float:
    enable_write_current = get_enable_write_current(data_dict)
    max_enable_current = get_max_enable_current(data_dict)
    write_temp = _calculate_channel_temperature(
        CRITICAL_TEMP, SUBSTRATE_TEMP, enable_write_current, max_enable_current
    )
    return write_temp


def get_write_current(data_dict: dict) -> float:
    if data_dict.get("write_current").shape[1] == 1:
        return filter_first(data_dict.get("write_current")) * 1e6
    if data_dict.get("write_current").shape[1] > 1:
        return data_dict.get("write_current")[0, 0] * 1e6


####################################################################################################


def get_operating_points(
    dict_list: list[dict], variable: Literal["write_current", "enable_write_current"]
) -> Tuple[list, list]:
    operating_points = []
    currents = []
    for i, data_dict in enumerate(dict_list):
        if variable == "write_current":
            currents.append(get_write_current(data_dict))
        if variable == "enable_write_current":
            currents.append(get_enable_write_current(data_dict))
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        nominal_opearting_point = np.mean([berargs[2], berargs[3]])
        inverting_operating_point = np.mean([berargs[0], berargs[1]])

        operating_points.append((nominal_opearting_point, inverting_operating_point))
    return currents, operating_points


def get_operating_margins(
    dict_list: list[dict], variable: Literal["write_current", "enable_write_current"]
) -> Tuple[list, list]:
    operating_margins = []
    currents = []
    for i, data_dict in enumerate(dict_list):
        currents.append(get_write_current(data_dict))
        bit_error_rate = get_bit_error_rate(data_dict)
        berarg = get_bit_error_rate_args(bit_error_rate)
        nominal_width = berarg[2] - berarg[1]
        inverting_width = berarg[0] - berarg[3]
        operating_margins.append((nominal_width, inverting_width))
    return currents, operating_margins


def get_bit_error_rate_args(bit_error_rate: np.ndarray) -> list:

    nominal_data = np.argwhere(bit_error_rate < 0.45)
    inverting_data = np.argwhere(bit_error_rate > 0.55)

    if len(inverting_data) > 0:
        pos_edge1 = inverting_data[0][0]
        neg_edge1 = inverting_data[-1][0]
    else:
        pos_edge1 = np.nan
        neg_edge1 = np.nan

    if len(nominal_data) > 0:
        neg_edge2 = nominal_data[0][0]
        pos_edge2 = nominal_data[-1][0]
    else:
        neg_edge2 = np.nan
        pos_edge2 = np.nan
    return [pos_edge1, neg_edge1, neg_edge2, pos_edge2]


def get_state_currents(
    read_currents: np.ndarray, bit_error_rate: np.ndarray
) -> Tuple[float, float]:

    state_idx = get_bit_error_rate_args(bit_error_rate)
    state0_current = np.nan
    state1_current = np.nan
    if state_idx[0] is not np.nan:
        state0_current = read_currents[state_idx[0]]
        state1_current = read_currents[state_idx[1]]
    if state_idx[2] is not np.nan:
        state0_current = read_currents[state_idx[3]]
        state1_current = read_currents[state_idx[2]]

    return state0_current, state1_current


####################################################################################################

####################################################################################################


def get_trace_data(
    data_dict: dict,
    trace_name: Literal["trace_chan_in", "trace_chan_out", "trace_enab"],
    trace_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if data_dict.get(trace_name).ndim == 2:
        x = data_dict[trace_name][0] * 1e6
        y = data_dict[trace_name][1] * 1e3
    else:
        x = data_dict[trace_name][0][:, trace_index] * 1e6
        y = data_dict[trace_name][1][:, trace_index] * 1e3
    return x, y


def save_directory_list(file_path: str, file_list: list[str]) -> None:
    with open(os.path.join(file_path, "data.txt"), "w") as f:
        for file_name in file_list:
            f.write(file_name + "\n")

    f.close()

    return


def import_directory(file_path: str) -> list[dict]:
    dict_list = []
    files = get_file_names(file_path)
    for file in files:
        data = sio.loadmat(os.path.join(file_path, file))
        dict_list.append(data)

    save_directory_list(file_path, files)
    return dict_list


def initialize_dict(array_size: tuple) -> dict:
    return {
        "write_current": np.zeros(array_size),
        "write_current_norm": np.zeros(array_size),
        "read_current": np.zeros(array_size),
        "read_current_norm": np.zeros(array_size),
        "slope": np.zeros(array_size),
        "y_intercept": np.zeros(array_size),
        "x_intercept": np.zeros(array_size),
        "resistance": np.zeros(array_size),
        "bit_error_rate": np.zeros(array_size),
        "max_critical_current": np.zeros(array_size),
        "enable_write_current": np.zeros(array_size),
        "enable_write_current_norm": np.zeros(array_size),
        "enable_read_current": np.zeros(array_size),
        "enable_read_current_norm": np.zeros(array_size),
        "enable_write_power": np.zeros(array_size),
        "enable_read_power": np.zeros(array_size),
    }


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.00), *zip(x, y), (x[-1], 0.00)]


def polygon_nominal(x: np.ndarray, y: np.ndarray) -> list:
    y = np.copy(y)
    y[y > 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def polygon_inverting(x: np.ndarray, y: np.ndarray) -> list:
    y = np.copy(y)
    y[y < 0.5] = 0.5
    return [(x[0], 0.5), *zip(x, y), (x[-1], 0.5)]


def process_cell(cell: dict, param_dict: dict, x: int, y: int) -> dict:
    param_dict["write_current"][y, x] = cell["write_current"] * 1e6
    param_dict["read_current"][y, x] = cell["read_current"] * 1e6
    param_dict["enable_write_current"][y, x] = cell["enable_write_current"] * 1e6
    param_dict["enable_read_current"][y, x] = cell["enable_read_current"] * 1e6
    param_dict["slope"][y, x] = cell["slope"]
    param_dict["y_intercept"][y, x] = cell["y_intercept"]
    param_dict["resistance"][y, x] = cell["resistance_cryo"]
    param_dict["bit_error_rate"][y, x] = cell.get("min_bit_error_rate", np.nan)
    param_dict["max_critical_current"][y, x] = (
        cell.get("max_critical_current", np.nan) * 1e6
    )
    if cell["y_intercept"] != 0:
        write_critical_current = htron_critical_current(
            cell["enable_write_current"] * 1e6, cell["slope"], cell["y_intercept"]
        )
        read_critical_current = htron_critical_current(
            cell["enable_read_current"] * 1e6, cell["slope"], cell["y_intercept"]
        )
        param_dict["x_intercept"][y, x] = -cell["y_intercept"] / cell["slope"]
        param_dict["write_current_norm"][y, x] = (
            cell["write_current"] * 1e6 / write_critical_current
        )
        param_dict["read_current_norm"][y, x] = (
            cell["read_current"] * 1e6 / read_critical_current
        )
        param_dict["enable_write_power"][y, x] = calculate_heater_power(
            cell["enable_write_current"], cell["resistance_cryo"]
        )
        param_dict["enable_write_current_norm"][y, x] = (
            cell["enable_write_current"] * 1e6 / param_dict["x_intercept"][y, x]
        )
        param_dict["enable_read_power"][y, x] = calculate_heater_power(
            cell["enable_read_current"], cell["resistance_cryo"]
        )
        param_dict["enable_read_current_norm"][y, x] = (
            cell["enable_read_current"] * 1e6 / param_dict["x_intercept"][y, x]
        )
    return param_dict


# def calculate_right_critical_currents(
#     enable_write_currents: np.ndarray, cell: str, width_ratio: float, iretrap: float
# ) -> np.ndarray:
#     channel_critical_current = (enable_write_currents * cell["slope"]) + cell[
#         "y_intercept"
#     ]
#     right_critical_current = channel_critical_current * (
#         1 + ((1 / width_ratio) * iretrap)
#     )
#     return right_critical_current


def plot_operating_points(ax: Axes, dict_list: list[dict]) -> Axes:
    currents, operating_points = get_operating_points(dict_list, "write_current")
    operating_points_array = np.array(operating_points)
    ax.plot(currents, operating_points_array[:, 0], label="Nominal Peak")
    ax.plot(currents, operating_points_array[:, 1], label="Inverting Peak")
    # ax.plot(currents, [x[0] for x in operating_points], label="Nominal Peak")
    # ax.plot(currents, [x[1] for x in operating_points], label="Inverting Peak")
    ax.legend()
    ax.set_ylabel("Peak Location ($\mu$A)")
    ax.set_xlabel("Write Current ($\mu$A)")

    return ax


def plot_operating_margins(ax: Axes, dict_list: list[dict]) -> Axes:
    currents, margins = get_operating_margins(dict_list, "write_current")
    ax.plot(currents, [x[0] for x in margins], label="Nominal Width")
    ax.plot(currents, [x[1] for x in margins], label="Inverting Width")
    ax.set_xlabel("Write Current ($\mu$A)")
    ax.set_ylabel("Width ($\mu$A)")
    ax.legend()
    return ax


# def plot_peak_distance(ax: Axes, data_dict: dict) -> Axes:
#     currents, operating_points = get_operating_points(data_dict, "write_current")
#     peak_distances = np.array([x[0] - x[1] for x in operating_points])
#     ax.plot(
#         currents,
#         peak_distances,
#         label="Peak Distance",
#         marker="o",
#         color="C4",
#     )
#     # plot a linear fit
#     plot_linear_fit(ax, currents[5:-1], peak_distances[5:-1])
#     ax.set_xlabel("Write Current ($\mu$A)")
#     ax.set_ylabel("Peak Distance ($\mu$A)")
#     ax.grid(True)

#     return ax


def plot_fill_between(ax, data_dict, color):
    # fill the area between 0.5 and the curve
    enable_write_currents = get_enable_write_currents(data_dict)
    bit_error_rate = get_bit_error_rate(data_dict)
    verts = polygon_nominal(enable_write_currents, bit_error_rate)
    poly = PolyCollection([verts], facecolors=color, alpha=0.6, edgecolors="k")
    ax.add_collection(poly)
    verts = polygon_inverting(enable_write_currents, bit_error_rate)
    poly = PolyCollection([verts], facecolors=color, alpha=0.6, edgecolors="k")
    ax.add_collection(poly)


def plot_enable_write_sweep_multiple(ax: Axes, dict_list: list[dict]) -> Axes:
    colors = CMAP(np.linspace(0, 1, len(dict_list)))
    for i, data_dict in enumerate(dict_list):
        plot_enable_write_sweep_single(ax, data_dict, color=colors[i])
        plot_fill_between(ax, data_dict, colors[i])

    ax2 = ax.twiny()
    write_temps = get_write_temperatures(data_dict)
    ax2.set_xlim([write_temps[0], write_temps[-1]])
    ax2.xaxis.set_major_locator(MultipleLocator(0.05))

    ax2.set_xlabel("Write Temperature (K)")
    ax.set_xlabel("Enable Write Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_yscale("log")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def plot_enable_write_sweep_single(
    ax: Axes,
    data_dict: dict,
    **kwargs,
) -> Axes:
    enable_write_currents = get_enable_write_currents(data_dict)
    bit_error_rate = get_bit_error_rate(data_dict)
    write_current = get_write_current(data_dict)
    ax.plot(
        enable_write_currents,
        bit_error_rate,
        label=f"$I_{{W}}$ = {write_current:.1f}$\mu$A",
        linewidth=2,
        **kwargs,
    )

    ax.set_xlim(enable_write_currents[0], enable_write_currents[-1])

    return ax


def plot_waterfall(ax: Axes3D, dict_list: list[dict]) -> Axes3D:
    colors = CMAP(np.linspace(0, 1, len(dict_list)))
    verts_list = []
    zlist = []

    for i, data_dict in enumerate(dict_list):
        enable_write_currents = get_enable_write_currents(data_dict)
        bit_error_rate = get_bit_error_rate(data_dict)
        write_current = get_write_current(data_dict)

        ax.plot(
            enable_write_currents,
            bit_error_rate,
            zs=write_current,
            zdir="y",
            color=colors[i],
            marker=".",
            markerfacecolor="k",
            markersize=5,
            linewidth=2,
        )
        zlist.append(write_current)
        verts = polygon_under_graph(enable_write_currents, bit_error_rate)
        verts_list.append(verts)

    poly = PolyCollection(verts_list, facecolors=colors, alpha=0.6, edgecolors="k")
    ax.add_collection3d(poly, zs=zlist, zdir="y")

    ax.set_xlabel("$I_{{EW}}$ ($\mu$A)", labelpad=10)
    ax.set_ylabel("$I_W$ ($\mu$A)", labelpad=70)
    ax.set_zlabel("BER", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=12, pad=5)

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)

    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])
    ax.set_ylim(10, zlist[-1])
    ax.set_yticks(zlist)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_box_aspect([0.5, 1, 0.2], zoom=0.8)
    ax.view_init(20, 35)
    # ax.view_init(25, 90)
    # ax.grid(False)
    return ax


def plot_linear_fit(ax: Axes, x: np.ndarray, y: np.ndarray) -> Axes:
    m, b = np.polyfit(x, y, 1)
    ax.plot(
        x,
        y,
        label="Peak Distance",
        marker="o",
        linestyle="None",
        markeredgecolor="k",
        markerfacecolor="None",
        markeredgewidth=2,
    )
    ax.text(
        np.mean([x[0], x[-1]]),
        np.mean([y[0], y[-1]]) * 1.1,
        f"y = {m:.2f}x + {b:.2f}",
        ha="right",
        va="bottom",
        fontsize=12,
    )
    # PLot the fit line
    x = np.array([x[0], x[-1]])
    ax.plot(x, m * x + b, "k--")
    return ax


def plot_channel_temperature(ax: plt.Axes, data_dict: dict, **kwargs) -> Axes:
    heater_current = data_dict.get("x")[0][:, 0] * 1e6
    max_enable_current = get_max_enable_current(data_dict)
    channel_temp = _calculate_channel_temperature(
        CRITICAL_TEMP, SUBSTRATE_TEMP, heater_current, max_enable_current
    )
    ax.plot(heater_current, channel_temp, **kwargs)
    ax.set_xlabel("Heater Current ($\mu$A)")
    ax.set_ylabel("Channel Temperature (K)")

    return ax


def plot_threshold(ax: Axes, start: int, end: int, threshold: float) -> Axes:
    ax.hlines(threshold, start, end, color="red", ls="-", lw=1)
    return ax


def plot_message(ax: Axes, message: str) -> Axes:
    axheight = ax.get_ylim()[1]
    for i, bit in enumerate(message):
        text = get_text_from_bit_v3(bit)
        ax.text(i + 0.5, axheight * 0.85, text, ha="center", va="center")

    return ax


def plot_trace_zoom(
    ax: Axes, x: np.ndarray, y: np.ndarray, start: float, end: float
) -> Axes:
    xzoom = x[(x > start) & (x < end)]
    yzoom = y[(x > start) & (x < end)]

    # smooth the yzoom data
    yzoom = np.convolve(yzoom, np.ones(20) / 20, mode="same")
    ax.plot(xzoom, 400 + yzoom * 10, color="red", ls="--", lw=1)
    ax.hlines(400, start, end, color="grey", ls="--", lw=1)

    return ax


def plot_chan_in(ax: Axes, data_dict: dict, trace_index: int) -> Axes:
    message = data_dict["bitmsg_channel"][0]
    x, y = get_trace_data(data_dict, "trace_chan_in", trace_index)
    ax.plot(x, y, color="#08519C")
    ax = plot_message(ax, message)

    plot_trace_zoom(x, y, 0.9, 2.1)
    plot_trace_zoom(x, y, 4.9, 6.1)

    ax.set_xticks(np.linspace(x[0], x[-1], 11))
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.set_xlabel("Time [$\mu$s]")
    ax.set_ylabel("Voltage [mV]")

    ax.grid(axis="x")
    return ax


def plot_chan_out(ax: Axes, data_dict: dict, trace_index: int) -> Axes:
    message = data_dict["bitmsg_channel"][0]
    x, y = get_trace_data(data_dict, "trace_chan_out", trace_index)

    ax.plot(x, y, color="#740F15")
    ax = plot_message(ax, message)

    plot_trace_zoom(x, y, 0.9, 2.1)
    plot_trace_zoom(x, y, 4.9, 6.1)

    ax.set_xticks(np.linspace(x[0], x[-1], 11))
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.set_xlabel("Time [$\mu$s]")
    ax.set_ylabel("Voltage [mV]")

    ax.grid(axis="x")
    return ax


def plot_enable(ax: Axes, data_dict: dict, trace_index: int):
    x, y = get_trace_data(data_dict, "trace_enable", trace_index)
    ax.plot(x, y, color="#DBB40C")

    ax.set_xticks(np.linspace(x[0], x[-1], 11))
    ax.set_xticklabels([f"{i:.1f}" for i in np.linspace(x[0], x[-1], 11)])
    ax.grid(axis="x")

    ax.set_xlabel("Time [$\mu$s]")
    ax.set_ylabel("Voltage [mV]")
    return ax


def plot_state_current_markers(ax: Axes, data_dict: dict, **kwargs) -> Axes:
    read_currents = get_read_currents(data_dict)
    bit_error_rate = get_bit_error_rate(data_dict)

    state0_current, state1_current = get_state_currents(read_currents, bit_error_rate)
    ax.plot(
        read_currents[read_currents == state0_current],
        bit_error_rate[read_currents == state0_current],
        marker="D",
        markeredgecolor="k",
        linewidth=1.5,
        label="_state0",
        **kwargs,
    )
    ax.plot(
        read_currents[read_currents == state1_current],
        bit_error_rate[read_currents == state1_current],
        marker="P",
        markeredgecolor="k",
        linewidth=1.5,
        label="_state1",
        **kwargs,
    )
    ax.plot(
        [
            read_currents[read_currents == state0_current],
            read_currents[read_currents == state1_current],
        ],
        [
            bit_error_rate[read_currents == state0_current],
            bit_error_rate[read_currents == state1_current],
        ],
        color="grey",
    )

    return ax


def plot_stack(
    axs: list[Axes],
    data_dict: list[dict],
    persistent_current: list[float],
) -> list[Axes]:
    for i in range(len(axs)):
        data_dict[i]["persistent_current"] = persistent_current[i]
        axs[i] = plot_analytical(axs[i], data_dict[i])
        axs[i].set_aspect("equal")
    return axs


def plot_read_sweep(
    ax: Axes,
    data_dict: dict,
    value_name: Literal["bit_error_rate", "write_0_read_1", "write_1_read_0"],
    variable_name: Literal[
        "enable_write_current",
        "read_width",
        "write_width",
        "write_current",
        "enable_read_current",
    ],
    **kwargs,
) -> Axes:
    write_temp = None
    label = None
    read_currents = get_read_currents(data_dict)

    if value_name == "bit_error_rate":
        value = get_bit_error_rate(data_dict)
    if value_name == "write_0_read_1":
        value = data_dict.get("write_0_read_1").flatten()
    if value_name == "write_1_read_0":
        value = data_dict.get("write_1_read_0").flatten()

    if variable_name == "write_current":
        variable = get_write_current(data_dict)
    if variable_name == "enable_write_current":
        variable = get_enable_write_current(data_dict)
        write_temp = get_write_temperature(data_dict)
        if write_temp is None:
            label = f"{variable:.2f}$\mu$A"
        else:
            label = f"{variable:.2f}$\mu$A, {write_temp:.2f}K"
    if variable_name == "read_width":
        variable = get_read_width(data_dict)
        label = f"{variable:.2f} pts "
    if variable_name == "write_width":
        variable = get_write_width(data_dict)
        label = f"{variable:.2f} pts "
    if variable_name == "enable_read_current":
        variable = get_enable_read_current(data_dict)

    ax.plot(
        read_currents,
        value,
        label=label,
        marker=".",
        markeredgecolor="k",
        **kwargs,
    )

    ax.set_ylim(0, 1)
    ax.set_title(f"{variable_name}")
    ax.legend(frameon=True, loc=2)
    ax.set_xlabel("Read Current [$\mu$A]")
    ax.set_ylabel("Normalized Bit Error Rate")
    return ax


def plot_read_sweep_array(
    ax: Axes, dict_list: list[dict], value_name: str, variable_name: str
) -> Axes:
    colors = CMAP(np.linspace(0, 1, len(dict_list)))
    for i, data_dict in enumerate(dict_list):
        plot_read_sweep(ax, data_dict, value_name, variable_name, color=colors[i])
        plot_bit_error_rate_args(ax, data_dict, color=colors[i])
        # plot_fill_between(ax, data_dict, colors[i])

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    return ax


def plot_read_delay(ax: Axes, dict_list: dict) -> Axes:
    colors = CMAP(np.linspace(0.2, 1, len(dict_list)))
    for i, data_dict in enumerate(dict_list):
        read_currents = get_read_currents(data_dict)
        bit_error_rate = get_bit_error_rate(data_dict)
        ax.plot(
            read_currents,
            bit_error_rate,
            label=f"+{i}$\mu$s",
            color=colors[i],
            marker=".",
            markeredgecolor="k",
        )
    ax.set_xlim(read_currents[0], read_currents[-1])
    ax.set_xlabel("Read Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    ax.grid(True)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    ax.set_yscale("log")
    return ax


def plot_bit_error_rate_args(ax: Axes, data_dict: dict, color) -> Axes:
    bit_error_rate = get_bit_error_rate(data_dict)
    berargs = get_bit_error_rate_args(bit_error_rate)

    read_current = get_read_currents(data_dict)
    for arg in berargs:
        if arg is not np.nan:
            ax.plot(
                read_current[arg],
                bit_error_rate[arg],
                marker="o",
                color=color,
            )
            ax.axvline(read_current[arg], color=color, linestyle="--")
    return ax


def plot_read_sweep_array_3d(ax: Axes3D, data_dict: dict) -> Axes3D:
    for key in data_dict.keys():
        ax = plot_read_sweep_3d(ax, data_dict[key])

    ax.xaxis.set_rotate_label(True)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(True)

    ax.set_zlim(0, 1)
    ax.set_zticks([0, 0.5, 1])
    ax.set_xlim(500, 950)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect([0.5, 1, 0.25], zoom=0.8)
    ax.view_init(20, -35)
    ax.grid(False)

    ax.set_position([0.0, 0.0, 1, 1])

    for child in ax.get_children():
        if isinstance(child, Rectangle):
            child.set_visible(False)
    return ax


def plot_read_sweep_3d(ax: Axes3D, data_dict: dict) -> Axes3D:
    cmap = plt.get_cmap("RdBu")
    colors = cmap(np.linspace(0, 1, 5))

    read_currents = get_bit_error_rate(data_dict)
    bit_error_rate = get_bit_error_rate(data_dict)
    enable_read_current = get_enable_read_current(data_dict)

    verts = polygon_nominal(read_currents, bit_error_rate)
    inv_verts = polygon_inverting(read_currents, bit_error_rate)

    poly = PolyCollection([verts], facecolors=colors[-1], alpha=0.6, edgecolors=None)
    poly_inv = PolyCollection(
        [inv_verts], facecolors=colors[0], alpha=0.6, edgecolors=None
    )
    ax.add_collection3d(poly, zs=[enable_read_current], zdir="y")
    ax.add_collection3d(poly_inv, zs=[enable_read_current], zdir="y")

    ax.set_xlabel("$I_R$ ($\mu$A)", labelpad=-6)
    ax.set_ylabel("$I_{{ER}}$ ($\mu$A)", labelpad=4)
    ax.set_zlabel("BER", labelpad=-6)
    ax.tick_params(axis="both", which="major", pad=-1)

    return ax


def plot_enable_write_sweep_grid(
    ax: Axes, dict_list: list[dict], save: bool = False
) -> None:

    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(dict_list)))
    for i, j in zip(["A", "B", "C", "D"], [2, 6, 7, 10]):
        ax[i] = plot_read_sweep(
            ax[i],
            dict_list[j],
            "bit_error_rate",
            "enable_write_current",
            color=colors[j],
        )
        ax[i].set_xlabel("Read Current ($\mu$A)")
        if i == "A":
            ax[i].set_ylabel("Bit Error Rate")

    ax["E"] = plot_state_currents(ax["E"], dict_list)
    return


def plot_bit_error_rate(ax: Axes, data_dict: dict) -> Axes:
    bit_error_rate = get_bit_error_rate(data_dict)
    measurement_param = data_dict.get("y")[0][:, 1] * 1e6

    # ax.plot(
    #     measurement_param,
    #     w0r1,
    #     color="#DBB40C",
    #     label="Write 0 Read 1",
    #     marker=".",
    # )
    # ax.plot(
    #     measurement_param,
    #     w1r0,
    #     color="#740F15",
    #     label="Write 1 Read 0",
    #     marker=".",
    # )
    ax.plot(
        measurement_param,
        bit_error_rate,
        color="#08519C",
        label="Total Bit Error Rate",
        marker=".",
    )
    ax.legend()
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Normalized\nBit Error Rate")
    return ax


def plot_voltage_trace(
    ax: Axes, time: np.ndarray, voltage: np.ndarray, **kwargs
) -> Axes:
    ax.plot(time, voltage, **kwargs)
    ax.set_xlim(time[0], time[-1])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_xticklabels([])
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="x", direction="in", which="both")
    ax.grid(axis="x", which="both")
    return ax


def plot_voltage_trace_stack(
    axs: List[Axes], data_dict: dict, trace_index: int = 0
) -> List[Axes]:
    colors = CMAP(np.linspace(0.2, 1, 3))
    colors = np.flipud(colors)
    if len(axs) != 3:
        raise ValueError("The number of axes must be 3.")

    chan_in_x, chan_in_y = get_trace_data(data_dict, "trace_chan_in", trace_index)
    chan_out_x, chan_out_y = get_trace_data(data_dict, "trace_chan_out", trace_index)
    enab_in_x, enab_in_y = get_trace_data(data_dict, "trace_enab", trace_index)

    bitmsg_channel = data_dict.get("bitmsg_channel")[0]
    bitmsg_enable = data_dict.get("bitmsg_enable")[0]

    plot_voltage_trace(axs[0], chan_in_x, chan_in_y, color=colors[0], label="Input")

    if bitmsg_enable[1] == "W" and bitmsg_channel[1] != "N":
        plot_trace_zoom(axs[0], chan_in_x, chan_in_y, 0.9, 2.1)
        plot_trace_zoom(axs[0], chan_in_x, chan_in_y, 4.9, 6.1)

    if bitmsg_enable[3] == "W" and bitmsg_channel[3] != "N":
        plot_trace_zoom(axs[0], chan_in_x, chan_in_y, 2.9, 4.1)
        plot_trace_zoom(axs[0], chan_in_x, chan_in_y, 6.9, 8.1)

    plot_voltage_trace(axs[1], enab_in_x, enab_in_y, color=colors[1], label="Enable")

    plot_voltage_trace(axs[2], chan_out_x, chan_out_y, color=colors[2], label="Output")

    axs[2].xaxis.set_major_locator(MultipleLocator(5))
    axs[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    axs[2].set_xlim([0, 10])

    fig = plt.gcf()
    fig.supylabel("Voltage [mV]")
    fig.supxlabel("Time [$\mu$s]")
    fig.subplots_adjust(hspace=0.0)

    return axs


def plot_hist_2axis(data_dict: dict, trace_index: int):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.0)
    w1r0 = data_dict["read_zero_top"][0][:, trace_index]
    w0r1 = data_dict["read_one_top"][0][:, trace_index]

    ax1.hist(w1r0, bins=100, alpha=1, color="#740F15", label="Read 1")
    ax1.hist(w0r1, bins=100, alpha=0.5, color="#DBB40C", label="Read 0")
    ax2.hist(w1r0, bins=100, alpha=1, color="#740F15", label="Read 1")
    ax2.hist(w0r1, bins=100, alpha=0.5, color="#DBB40C", label="Read 0")

    ax2.set_ylim([0, 10])
    ax2.set_yticks([0, 5, 10])

    ax1.set_ylim([10, 10000])

    fig.supylabel("Count")
    fig.supxlabel("Voltage [mV]")
    plt.show()


def plot_voltage_hist(ax: Axes, voltage: np.ndarray, **kwargs) -> Axes:
    ax.hist(voltage, bins=77, **kwargs)

    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylabel("Counts")
    ax.set_xlabel("Voltage [mV]")

    return ax


def plot_text_labels(
    ax: Axes, xloc: np.ndarray, yloc: np.ndarray, ztotal: np.ndarray, log: bool
) -> Axes:
    for x, y in zip(xloc, yloc):
        text = f"{ztotal[y, x]:.2f}"
        txt_color = "black"
        if ztotal[y, x] > (0.8 * max(ztotal.flatten())):
            txt_color = "white"
        if log:
            text = f"{ztotal[y, x]:.1e}"
            txt_color = "black"

        ax.text(
            x,
            y,
            text,
            color=txt_color,
            backgroundcolor="none",
            ha="center",
            va="center",
            weight="bold",
        )

    return ax


def plot_array(
    ax: Axes,
    xloc: np.ndarray,
    yloc: np.ndarray,
    ztotal: np.ndarray,
    title: str = None,
    log: bool = False,
    reverse: bool = False,
    cmap: plt.cm = None,
) -> Axes:
    if cmap is None:
        cmap = plt.get_cmap("viridis")
    if reverse:
        cmap = cmap.reversed()

    if log:
        ax.matshow(ztotal, cmap=cmap, norm=LogNorm(vmin=1e-6, vmax=1e-2))
    else:
        ax.matshow(ztotal, cmap=cmap)

    if title:
        ax.set_title(title)

    ax.set_xticks(range(4), ["A", "B", "C", "D"])
    ax.set_yticks(range(4), ["1", "2", "3", "4"])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="both", length=0)
    ax = plot_text_labels(ax, xloc, yloc, ztotal, log)

    return ax


def plot_array_3d(
    xloc, yloc, ztotal, title=None, log=False, norm=False, reverse=False, ax=None
):
    if ax is None:
        fig, ax = plt.subplots()

    cmap = plt.cm.get_cmap("viridis")
    if reverse:
        cmap = plt.cm.get_cmap("viridis").reversed()

    ax.bar3d(xloc, yloc, 0, 1, 1, ztotal.flatten(), shade=True)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(range(4), ["A", "B", "C", "D"])
    ax.set_yticks(range(4), ["1", "2", "3", "4"])
    ax.set_zlim(0, np.nanmax(ztotal))
    ax.patch.set_visible(False)

    ax.tick_params(axis="both", which="major", labelsize=6, pad=0)
    # ax = plot_text_labels(xloc, yloc, ztotal, log, ax=ax)

    return ax


def plot_normalization(
    ax: Axes,
    write_current_norm: np.ndarray,
    read_current_norm: np.ndarray,
    enable_write_current: np.ndarray,
    enable_read_current: np.ndarray,
) -> Axes:
    # remove NaN from arrays
    write_current_norm = write_current_norm[~np.isnan(write_current_norm)]
    read_current_norm = read_current_norm[~np.isnan(read_current_norm)]
    enable_write_current = enable_write_current[~np.isnan(enable_write_current)]
    enable_read_current = enable_read_current[~np.isnan(enable_read_current)]

    # remove zeros from arrays
    write_current_norm = write_current_norm[write_current_norm != 0]
    read_current_norm = read_current_norm[read_current_norm != 0]
    enable_write_current = enable_write_current[enable_write_current != 0]
    enable_read_current = enable_read_current[enable_read_current != 0]

    ax.boxplot(write_current_norm.flatten(), positions=[0], widths=0.5)
    ax.boxplot(read_current_norm.flatten(), positions=[1], widths=0.5)
    ax.boxplot(enable_write_current.flatten(), positions=[2], widths=0.5)
    ax.boxplot(enable_read_current.flatten(), positions=[3], widths=0.5)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        [
            "Write Current",
            "Read Current",
            "Enable\nWrite Current",
            "Enable\nRead Current",
        ]
    )
    ax.set_xlabel("Input Type")
    ax.set_xticks(rotation=45)
    ax.set_ylabel("Normalized Current")
    ax.set_yticks(np.linspace(0, 1, 11))
    return ax


def plot_analytical(ax: Axes, data_dict: dict) -> Axes:
    color_map = plt.get_cmap("RdBu")

    read_current_dict = calculate_read_currents(data_dict)
    inv_region = np.where(
        np.maximum(
            data_dict["read_currents_mesh"]
            - read_current_dict["one_state_currents_inv"],
            read_current_dict["zero_state_currents_inv"]
            - data_dict["read_currents_mesh"],
        )
        <= 0,
        data_dict["read_currents_mesh"],
        np.nan,
    )
    nominal_region = np.where(
        np.maximum(
            data_dict["read_currents_mesh"] - read_current_dict["zero_state_currents"],
            read_current_dict["one_state_currents"] - data_dict["read_currents_mesh"],
        )
        <= 0,
        data_dict["read_currents_mesh"],
        np.nan,
    )
    ax.pcolormesh(
        data_dict["right_critical_currents_mesh"],
        data_dict["read_currents_mesh"],
        nominal_region,
        cmap=color_map,
        vmin=-1000,
        vmax=1000,
        zorder=0,
    )
    ax.pcolormesh(
        data_dict["right_critical_currents_mesh"],
        data_dict["read_currents_mesh"],
        -1 * inv_region,
        cmap=color_map,
        vmin=-1000,
        vmax=1000,
        zorder=0,
    )
    read_current_dict["nominal"] = nominal_region
    read_current_dict["inverting"] = inv_region
    return ax


def plot_cell_param(ax: Axes, param: str) -> Axes:
    param_array = np.array([CELLS[cell][param] for cell in CELLS]).reshape(4, 4)

    plot_array(
        ax,
        np.arange(4),
        np.arange(4),
        param_array * 1e6,
        f"Cell {param}",
        log=False,
        norm=False,
        reverse=False,
    )
    return ax


def plot_fit(ax: Axes, xfit: np.ndarray, yfit: np.ndarray) -> Axes:
    z = np.polyfit(xfit, yfit, 1)
    p = np.poly1d(z)
    x_intercept = -z[1] / z[0]
    ax.scatter(xfit, yfit, color="#08519C")
    xplot = np.linspace(0, x_intercept, 10)
    ax.plot(xplot, p(xplot), "--", color="#740F15")
    ax.text(
        0.1,
        0.1,
        f"{p[1]:.3f}x + {p[0]:.3f}\n$x_{{int}}$ = {x_intercept:.2f}",
        fontsize=12,
        color="red",
        backgroundcolor="white",
        transform=ax.transAxes,
    )

    return ax


# def construct_array(data_dict: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     x = data_dict.get("x")[0][:, 1] * 1e6
#     y = data_dict.get("y")[0][:, 0] * 1e6
#     w0r1 = 100 - data_dict.get("write_0_read_1")[0].flatten()
#     w1r0 = data_dict.get("write_1_read_0")[0].flatten()
#     z = w1r0 + w0r1
#     ztotal = z.reshape((len(y), len(x)), order="F")
#     return x, y, ztotal


# def plot_enable_current_relation(
#     ax: Axes, data_dict: dict, parameter_z: str = "total_switches_norm"
# ) -> Axes:
#     x, y, ztotal = build_array(data_dict, parameter_z)
#     dx, dy = np.diff(x)[0], np.diff(y)[0]
#     xfit, yfit = get_fitting_points(x, y, ztotal)

#     ax.scatter(xfit, yfit)

#     # Plot a fit line to the scatter points
#     plot_fit(ax, xfit, yfit)

#     ax.matshow(
#         ztotal,
#         extent=[
#             (-0.5 * dx + x[0]),
#             (0.5 * dx + x[-1]),
#             (-0.5 * dy + y[0]),
#             (0.5 * dy + y[-1]),
#         ],
#         aspect="auto",
#         origin="lower",
#         cmap="viridis",
#     )
#     ax.xaxis.tick_bottom()
#     ax.xaxis.set_major_locator(MaxNLocator(len(x)))
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
#     ax.yaxis.set_major_locator(MultipleLocator(50))

#     return ztotal


def find_max_critical_current(data):
    _, y, ztotal = build_array(data, "bit_error_rate")

    # Find the maximum critical current using np.diff
    diff = np.diff(ztotal)
    mid_idx = np.where(diff == np.max(diff))

    return np.mean(y[mid_idx])


# def plot_read_temp_sweep_C3():
#     fig, axs = plt.subplots(2, 2, figsize=(12, 6))

#     plot_write_sweep(axs[0, 0], "write_current_sweep_C3_2")
#     plot_write_sweep(axs[0, 1], "write_current_sweep_C3_3")
#     plot_write_sweep(axs[1, 0], "write_current_sweep_C3_4")
#     plot_write_sweep(axs[1, 1], "write_current_sweep_C3")
#     axs[1, 1].legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")
#     fig.subplots_adjust(hspace=0.5, wspace=0.3)


# def plot_write_sweep(ax, data_directory: str):
#     dict_list = import_directory(data_directory)
#     colors = plt.cm.viridis(np.linspace(0, 1, len(dict_list)))
#     ax2 = ax.twinx()

#     for data in [dict_list[-7]]:

#         cell = data.get("cell")[0]
#         max_enable_current = filter_first(
#             data["CELLS"][cell][0][0]["x_intercept"]
#         ).flatten()[0]
#         enable_read_current = filter_first(data.get("enable_read_current")) * 1e6
#         enable_write_current = filter_first(data.get("enable_write_current")) * 1e6

#         x, y, ztotal = build_array(data, "bit_error_rate")
#         _, _, zswitch = build_array(data, "total_switches_norm")

#         enable_write_temp = calculate_channel_temperature(
#             SUBSTRATE_TEMP, CRITICAL_TEMP, enable_write_current, max_enable_current
#         ).flatten()[0]
#         ax.plot(
#             y,
#             ztotal,
#             label=f"enable_write_current = {enable_write_current:.0f} $\mu$A, Write Temperature: {enable_write_temp:.2f} K",
#             color=colors[dict_list.index(data)],
#         )
#         ax2.plot(
#             y,
#             zswitch,
#             label="_",
#             color="grey",
#             linewidth=0.5,
#             linestyle=":",
#         )

#         write_Ic = calculate_critical_current(enable_read_current, CELLS[cell])
#         print(f"Write Ic: {write_Ic}, enable_write_current: {enable_write_current}")
#         ax.hlines(
#             write_Ic, ax.get_xlim()[0], ax.get_xlim()[1], linestyle="--", color="k"
#         )

#         enable_read_temp = calculate_channel_temperature(
#             SUBSTRATE_TEMP, CRITICAL_TEMP, enable_read_current, max_enable_current
#         ).flatten()[0]
#         enable_write_temp = calculate_channel_temperature(
#             SUBSTRATE_TEMP, CRITICAL_TEMP, enable_write_current, max_enable_current
#         ).flatten()[0]

#     ax.set_title(
#         f"Cell {cell} - Read Temperature: {enable_read_temp:.2f}, Write Temperature: {enable_write_temp:.2f} K"
#     )
#     ax.legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")
#     ax.set_ylim([0, 1])
#     # ax.set_xlim([500, 900])
#     ax.set_xlabel("Write Current ($\mu$A)")
#     ax.set_ylabel("Bit Error Rate")
#     ax2.set_ylim([0, 1])
#     ax2.set_ylabel("Switching Probability")
#     ax2.yaxis.set_major_locator(MultipleLocator(0.5))
#     ax.yaxis.set_major_locator(MultipleLocator(0.5))
#     # calculate_critical_current(enable_read_current, data["CELLS"][cell][0][0])


def plot_state_separation(ax: Axes, dict_list: list[dict]) -> Axes:
    state_separation = []
    enable_write_currents_list = []
    for i, data_dict in enumerate(dict_list):
        enable_write_current = get_enable_write_current(data_dict)
        read_currents = get_read_currents(data_dict)
        bit_error_rate = get_bit_error_rate(data_dict)
        state0_current, state1_current = get_state_currents(
            read_currents, bit_error_rate
        )
        state_separation.append(state1_current - state0_current)
        enable_write_currents_list.append(enable_write_current)

    ax.bar(
        enable_write_currents_list,
        state_separation,
        width=2.5,
    )
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    ax.grid(True, axis="both", which="both")
    ax.set_xlabel("Enable Write Current ($\mu$A)")
    ax.set_ylabel("Diff. Between State Currents ($\mu$A)")
    return ax


def plot_state_currents(ax: Axes, dict_list: list[dict]) -> Axes:
    cmap = plt.get_cmap("Spectral")
    colors = cmap(np.linspace(0, 1, len(dict_list)))
    enable_write_currents_list = []
    state0_currents = []
    state1_currents = []
    for i, data_dict in enumerate(dict_list):
        read_currents = get_read_currents(data_dict)
        bit_error_rate = get_bit_error_rate(data_dict)
        enable_write_currents = get_enable_write_current(data_dict)
        state0_current, state1_current = get_state_currents(
            read_currents, bit_error_rate
        )
        enable_write_currents_list.append(enable_write_currents)
        state0_currents.append(state0_current)
        state1_currents.append(state1_current)

    sort_arg = np.argsort(enable_write_currents_list)

    ax.plot(
        np.array(enable_write_currents_list)[sort_arg],
        np.array(state0_currents)[sort_arg],
        label="State 0",
        marker="D",
        color="grey",
        markeredgecolor="k",
        zorder=0,
    )
    ax.plot(
        np.array(enable_write_currents_list)[sort_arg],
        np.array(state1_currents)[sort_arg],
        label="State 1",
        marker="P",
        color="grey",
        markeredgecolor="k",
        markersize=10,
        zorder=0,
    )
    ax.set_xlabel("Enable Write Current ($\mu$A)")
    ax.set_ylabel("State Current ($\mu$A)")
    ax.grid(True, which="both", axis="both")
    ax.legend(frameon=False)

    return ax


def plot_read_temp_sweep_C3():
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    plot_write_sweep(axs[0, 0], "write_current_sweep_C3_2")
    plot_write_sweep(axs[0, 1], "write_current_sweep_C3_3")
    plot_write_sweep(axs[1, 0], "write_current_sweep_C3_4")
    plot_write_sweep(axs[1, 1], "write_current_sweep_C3")
    axs[1, 1].legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")
    fig.subplots_adjust(hspace=0.5, wspace=0.3)


def plot_slice(
    ax: Axes, data_dict: dict, parameter_z: str = "bit_error_rate", **kwargs
) -> Axes:
    x, y, ztotal = build_array(data_dict, parameter_z)
    if len(x) == 1:
        x = y
    ax.plot(x, ztotal, **kwargs)

    return ax


def plot_iv_curve(ax: Axes, data_dict: dict, **kwargs) -> Axes:
    time = data_dict.get("trace")[0, :]
    voltage = data_dict.get("trace")[1, :]

    M = int(np.round(len(voltage), -2))
    currentQuart = np.linspace(0, data_dict["vpp"] / 2 / 10e3, M // 4)
    current = np.concatenate(
        [-currentQuart, np.flip(-currentQuart), currentQuart, np.flip(currentQuart)]
    )

    if len(voltage) > M:
        voltage = voltage[:M]
        time = time[:M]
    else:
        voltage = np.concatenate([voltage, np.zeros(M - len(voltage))])
        time = np.concatenate([time, np.zeros(M - len(time))])

    ax.plot(voltage, current.flatten() * 1e6, **kwargs)
    return ax


def plot_iv(ax: Axes, dict_list: list, save: bool = False) -> Axes:
    colors = plt.cm.coolwarm(np.linspace(0, 1, int(len(dict_list) / 2) + 1))
    colors = np.flipud(colors)
    for i, data in enumerate(dict_list):
        heater_current = np.abs(data["heater_current"].flatten()[0] * 1e6)
        ax = plot_iv_curve(
            ax, data, color=colors[i], zorder=-i, label=f"{heater_current:.0f} µA"
        )
        if i == 10:
            break
    ax.set_ylim([-500, 500])
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [µA]", labelpad=-1)
    ax.tick_params(direction="in", top=True, right=True)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.legend(frameon=False, handlelength=0.5, labelspacing=0.1)

    if save:
        plt.savefig("iv_curve.pdf", bbox_inches="tight")

    return ax


def plot_critical_currents(ax: Axes, dict_list: list) -> Axes:
    critical_currents, critical_currents_std = get_critical_currents(dict_list)
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(dict_list)))
    heater_currents = [data["heater_current"].flatten() * 1e6 for data in dict_list]
    ax.errorbar(
        heater_currents,
        critical_currents,
        yerr=critical_currents_std,
        fmt="o",
        markersize=3,
        color=cmap[0, :],
    )
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel("Heater Current [µA]")
    ax.set_ylabel("Critical Current [µA]")
    ax.set_ylim([0, 400])
    return ax


def plot_critical_currents_abs(ax: Axes, dict_list: list, save: bool = False) -> Axes:
    critical_currents, critical_currents_std = get_critical_currents(dict_list)

    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(dict_list)))
    heater_currents = np.array(
        [data["heater_current"].flatten() * 1e6 for data in dict_list]
    ).flatten()
    positive_critical_currents = np.where(
        heater_currents > 0, critical_currents, np.nan
    )
    negative_critical_currents = np.where(
        heater_currents < 0, critical_currents, np.nan
    )
    ax.plot(
        np.abs(heater_currents),
        positive_critical_currents,
        "o--",
        color=cmap[0, :],
        label="$+I_{{h}}$",
        linewidth=0.5,
        markersize=0.5,
        markerfacecolor=cmap[0, :],
    )
    ax.plot(
        np.abs(heater_currents),
        negative_critical_currents,
        "o--",
        color=cmap[-5, :],
        label="$-I_{{h}}$",
        linewidth=0.5,
        markersize=0.5,
        markerfacecolor=cmap[-5, :],
    )
    ax.fill_between(
        np.abs(heater_currents),
        (positive_critical_currents + critical_currents_std),
        (positive_critical_currents - critical_currents_std),
        color=cmap[0, :],
        alpha=0.3,
        edgecolor="none",
    )
    ax.fill_between(
        np.abs(heater_currents),
        (negative_critical_currents + critical_currents_std),
        (negative_critical_currents - critical_currents_std),
        color=cmap[-5, :],
        alpha=0.3,
        edgecolor="none",
    )
    ax.tick_params(direction="in", top=True, right=True, bottom=True, left=True)
    ax.set_xlabel("$|I_{{h}}|$[$\mu$A]")

    ax.set_ylabel("$I_{{C}}$ [$\mu$A]", labelpad=-1)
    ax.set_ylim([0, 400])
    ax.set_xlim([0, 500])
    ax.legend(frameon=False, loc="upper left", handlelength=0.5, labelspacing=0.1)

    ax2 = ax.twinx()
    temp = _calculate_channel_temperature(1.3, 12.3, np.abs(heater_currents), 500)
    ax2.plot(np.abs(heater_currents), temp, color="black", linewidth=0.5)
    ax2.set_ylim([0, 13])
    ax2.set_ylabel("Temperature [K]")
    ax2.hlines([1.3, 12.3], 0, 500, color="black", linestyle="--", linewidth=0.5)

    if save:
        plt.savefig("critical_currents_abs.pdf", bbox_inches="tight")

    return ax


def plot_critical_currents_inset(ax: Axes, dict_list: list, save: bool = False) -> Axes:
    ax = plot_iv(ax, dict_list)
    fig = plt.gcf()
    ax_inset = fig.add_axes([0.62, 0.25, 0.3125, 0.25])
    ax_inset = plot_critical_currents_abs(ax_inset, dict_list)
    ax_inset.xaxis.tick_top()
    ax_inset.tick_params(direction="in", top=True, right=True, bottom=True, left=True)

    ax_inset.xaxis.set_label_position("top")
    ax_inset.xaxis.set_major_locator(MultipleLocator(0.1))

    if save:
        plt.savefig("critical_currents_inset.pdf", bbox_inches="tight")

    return ax


def plot_combined_figure(ax: Axes, dict_list: list, save: bool = False) -> Axes:
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")
    ax[1, 0].axis("off")
    ax[1, 1] = plot_iv(ax[1, 1], dict_list)
    ax[1, 2] = plot_critical_currents_abs(ax[1, 2], dict_list)

    if save:
        plt.subplots_adjust(wspace=0.3)
        plt.savefig("iv_curve_combined.pdf", bbox_inches="tight")

    return ax


def plot_trace_averaged(ax: Axes, data_dict: dict, trace_name: str, **kwargs) -> Axes:
    ax.plot(
        (data_dict[trace_name][0, :] - data_dict[trace_name][0, 0]) * 1e9,
        data_dict[trace_name][1, :] * 1e3,
        **kwargs,
    )
    return ax


def plot_hist(ax: Axes, data_dict: dict) -> Axes:
    ax.hist(
        data_dict["read_zero_top"][0, :],
        log=True,
        range=(0.2, 0.6),
        bins=100,
        label="Read 0",
        color="#1966ff",
    )
    ax.hist(
        data_dict["read_one_top"][0, :],
        log=True,
        range=(0.2, 0.6),
        bins=100,
        label="Read 1",
        color="#ff1423",
    )
    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Counts")
    ax.legend()
    return ax


def plot_bitstream(ax: Axes, data_dict: dict, trace_name: str) -> Axes:
    x, y = get_trace_data(
        data_dict,
        trace_name,
    )
    ax.plot(
        data_dict[trace_name][0, :] * 1e6,
        data_dict[trace_name][1, :],
        label=trace_name,
    )
    plot_message(ax, data_dict["bitmsg_channel"][0])
    return ax


def plot_delay(ax: Axes, dict_list: list[dict]) -> Axes:
    bers = []
    for i in range(4):
        bers.append(get_bit_error_rate(dict_list[i]))

    ax.plot([1, 2, 3, 4], bers, label="bit_error_rate", marker="o", color="#345F90")
    ax.set_xlabel("Delay [$\mu$s]")
    ax.set_ylabel("BER")

    return ax


def create_combined_plot(data_dict: dict, save=False):
    fig = plt.figure(figsize=(7, 3.5))
    subfigs = fig.subfigures(
        2,
        3,
        width_ratios=[0.3, 0.7, 0.3],
        height_ratios=[0.4, 0.6],
        wspace=-0.1,
        hspace=0.0,
    )
    ax = subfigs[1, 0].subplots(4, 1)
    subfigs[1, 0].subplots_adjust(hspace=0.0)
    subfigs[1, 0].supylabel("Voltage [V]", x=-0.05)
    subfigs[1, 0].supxlabel("Time [ns]")
    plot_trace_averaged(ax[0], data_dict[4], "trace_write_avg", "#345F90")
    plot_trace_averaged(ax[1], data_dict[4], "trace_ewrite_avg", "#345F90")
    plot_trace_averaged(ax[2], data_dict[4], "trace_read0_avg", "#345F90")
    plot_trace_averaged(ax[2], data_dict[4], "trace_read1_avg", "#B3252C")
    plot_trace_averaged(ax[3], data_dict[4], "trace_eread_avg", "#345F90")
    ax[2].legend(["Read 0", "Read 1"], frameon=False)
    axhist = subfigs[0, 2].add_subplot()
    subfigs[0, 2].subplots_adjust(hspace=0.0)
    plot_hist(axhist, data_dict[3])

    axbits = subfigs[1, 1].subplots(4, 1)
    plot_bitstream(axbits[0], data_dict[5], trace_name="trace_chan_out")
    plot_bitstream(axbits[1], data_dict[6], trace_name="trace_chan_out")
    plot_bitstream(axbits[2], data_dict[7], trace_name="trace_chan_out")
    plot_bitstream(axbits[3], data_dict[8], trace_name="trace_chan_out")
    subfigs[1, 1].supylabel("Voltage [V]", x=0.05)
    subfigs[1, 1].supxlabel("Time [$\mu$s]")

    axdelay = subfigs[1, 2].add_subplot()
    plot_delay(axdelay, data_dict)

    fig.patch.set_visible(False)
    if save:
        plt.savefig("delay_plotting.pdf", bbox_inches="tight")
    plt.show()


def create_combined_plot_v2(data_dict: dict, save=False):
    fig = plt.figure(figsize=(7.087, 3.543))
    subfigs = fig.subfigures(
        2,
        3,
    )
    ax = subfigs[1, 1].subplots(2, 1)
    # subfigs[1, 1].subplots_adjust(hspace=0.0)
    subfigs[1, 1].supylabel("Voltage [V]", x=-0.05)
    # subfigs[1, 1].supxlabel("Time [ns]")
    ax2 = ax[0].twinx()
    ax3 = ax[1].twinx()
    plot_trace_averaged(
        ax[0], data_dict[4], "trace_write_avg", color="#293689", label="Write"
    )
    plot_trace_averaged(
        ax2, data_dict[4], "trace_ewrite_avg", color="#ff1423", label="Enable Write"
    )
    plot_trace_averaged(
        ax[1], data_dict[4], "trace_read0_avg", color="#1966ff", label="Read 0"
    )
    plot_trace_averaged(
        ax[1],
        data_dict[4],
        "trace_read1_avg",
        color="#ff14f0",
        linestyle="--",
        label="Read 1",
    )
    plot_trace_averaged(
        ax3, data_dict[4], "trace_eread_avg", color="#ff1423", label="Enable Read"
    )
    ax[1].legend(["Read 0", "Read 1"], frameon=False)
    ax[1].set_xlabel("Time [$\mu$s]")
    ax2.set_ylabel("[V]")

    axfit = subfigs[1, 2].add_subplot()
    # plot_all_cells(axfit)
    fig.patch.set_visible(False)
    if save:
        plt.savefig("delay_plotting_v2.pdf", bbox_inches="tight")
    plt.show()


def create_combined_plot_v3(data_dict: dict, save=False):
    fig = plt.figure(figsize=(6.264, 2))
    ax_dict = fig.subplot_mosaic("AC;BC")
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()
    plot_trace_averaged(
        ax_dict["A"], data_dict[4], "trace_write_avg", color="#293689", label="Write"
    )
    plot_trace_averaged(
        ax2, data_dict[4], "trace_ewrite_avg", color="#ff1423", label="Enable Write"
    )
    plot_trace_averaged(
        ax_dict["B"], data_dict[4], "trace_read0_avg", color="#1966ff", label="Read 0"
    )
    plot_trace_averaged(
        ax_dict["B"],
        data_dict[4],
        "trace_read1_avg",
        color="#ff14f0",
        linestyle="--",
        label="Read 1",
    )
    plot_trace_averaged(
        ax3, data_dict[4], "trace_eread_avg", color="#ff1423", label="Enable Read"
    )
    ax_dict["A"].legend(loc="upper left")
    ax_dict["A"].set_ylabel("[mV]")
    ax2.legend()
    ax2.set_ylabel("[mV]")
    ax3.legend()
    ax3.set_ylabel("[mV]")
    ax_dict["B"].set_xlabel("Time [$\mu$s]")
    ax_dict["B"].set_ylabel("[mV]")
    ax_dict["B"].legend(loc="upper left")

    fig.subplots_adjust(wspace=0.45)
    plot_enable_current_relation(ax_dict["C"], [data_dict])
    if save:
        plt.savefig("delay_plotting_v2.pdf", bbox_inches="tight")
    plt.show()


def create_combined_plot_v4(data_dict: dict, save=False):
    fig = plt.figure(figsize=(6.264, 2))
    ax_dict = fig.subplot_mosaic("AC;BC")
    ax2 = ax_dict["A"].twinx()
    ax3 = ax_dict["B"].twinx()
    plot_trace_averaged(
        ax_dict["A"], data_dict[4], "trace_write_avg", color="#293689", label="Write"
    )
    plot_trace_averaged(
        ax2, data_dict[4], "trace_ewrite_avg", color="#ff1423", label="Enable Write"
    )
    plot_trace_averaged(
        ax_dict["B"], data_dict[4], "trace_read0_avg", color="#1966ff", label="Read 0"
    )
    plot_trace_averaged(
        ax_dict["B"],
        data_dict[4],
        "trace_read1_avg",
        color="#ff14f0",
        linestyle="--",
        label="Read 1",
    )
    plot_trace_averaged(
        ax3, data_dict[4], "trace_eread_avg", color="#ff1423", label="Enable Read"
    )
    ax_dict["A"].legend(loc="upper left")
    ax_dict["A"].set_ylabel("[mV]")
    ax2.legend()
    ax2.set_ylabel("[mV]")
    ax3.legend()
    ax3.set_ylabel("[mV]")
    ax_dict["B"].set_xlabel("Time [$\mu$s]")
    ax_dict["B"].set_ylabel("[mV]")
    ax_dict["B"].legend(loc="upper left")

    fig.subplots_adjust(wspace=0.45)
    plot_hist(ax_dict["C"], data_dict[3])

    if save:
        plt.savefig("delay_plotting_v2.pdf", bbox_inches="tight")
    plt.show()


def plot_grid(axs, dict_list):
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for dict in dict_list:
        cell = dict.get("cell")[0]

        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1
        x = dict["x"][0]
        y = dict["y"][0]
        ztotal = dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)
        # xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.9)
        axs[row, column].plot(
            xfit, yfit, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )

        xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.75)
        plot_fit(
            axs[row, column],
            xfit,
            yfit,
            # color=colors[column],
            # marker=markers[row],
            # markeredgecolor="k",
        )
        enable_read_current = CELLS[cell].get("enable_read_current") * 1e6
        enable_write_current = CELLS[cell].get("enable_write_current") * 1e6
        axs[row, column].vlines(
            [enable_write_current],
            *axs[row, column].get_ylim(),
            linestyle="--",
            color="grey",
            label="Enable Write Current",
        )
        axs[row, column].vlines(
            [enable_read_current],
            *axs[row, column].get_ylim(),
            linestyle="--",
            color="r",
            label="Enable Read Current",
        )

        axs[row, column].legend(loc="upper right")
        axs[row, column].set_xlim(0, 600)
        axs[row, column].set_ylim(0, 1000)
        # axs[row, column].set_aspect("equal")
    axs[-1, 0].set_xlabel("Enable Current ($\mu$A)")
    axs[-1, 0].set_ylabel("Critical Current ($\mu$A)")
    return axs


def plot_row(axs, dict_list):
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for data_dict in dict_list:
        cell = get_current_cell(data_dict)

        column, row = convert_cell_to_coordinates(cell)
        x = data_dict["x"][0]
        y = data_dict["y"][0]
        ztotal = data_dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)

        axs[row].plot(
            xfit, yfit, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )

        axs[row].legend(loc="lower left")
        axs[row].set_xlim(0, 500)
        axs[row].set_ylim(0, 1000)
    axs[0].set_xlabel("Enable Current ($\mu$A)")
    axs[0].set_ylabel("Critical Current ($\mu$A)")
    return axs


def plot_column(axs, dict_list):
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for data_dict in dict_list:
        cell = get_current_cell(data_dict)

        column, row = convert_cell_to_coordinates(cell)
        x = data_dict["x"][0]
        y = data_dict["y"][0]
        ztotal = data_dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)
        # xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.9)
        axs[column].plot(
            xfit, yfit, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )

        axs[column].legend(loc="lower left")
        axs[column].set_xlim(0, 500)
        axs[column].set_ylim(0, 1000)
    axs[0].set_xlabel("Enable Current ($\mu$A)")
    axs[0].set_ylabel("Critical Current ($\mu$A)")
    return axs


def plot_full_grid(axs, dict_list):
    plot_grid(axs[1:5, 0:4], dict_list)

    plot_row(axs[0, 0:4], dict_list)

    plot_column(axs[1:5, 4], dict_list)
    axs[0, 4].axis("off")
    axs[4, 0].set_xlabel("Enable Current ($\mu$A)")
    axs[4, 0].set_ylabel("Critical Current ($\mu$A)")

    return axs


def plot_enable_current_relation(ax: Axes, dict_list: list[dict]) -> Axes:
    colors = plt.cm.RdBu(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for data_dict in dict_list:
        cell = get_current_cell(data_dict)
        column, row = convert_cell_to_coordinates(cell)
        x, y, ztotal = build_array(data_dict, "total_switches_norm")
        xfit, yfit = get_fitting_points(x, y, ztotal)
        ax.plot(xfit, yfit, label=f"{cell}", color=colors[column], marker=markers[row])

        ax.set_xlabel("Enable Current ($\mu$A)")
        ax.set_ylabel("Critical Current ($\mu$A)")
        ax.legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")
    return ax


def plot_enable_write_temperature(ax: Axes) -> Axes:
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for cell in CELLS.keys():
        xint = CELLS[cell].get("x_intercept")
        x = get_enable_write_current(CELLS[cell])
        temp = calculate_channel_temperature(SUBSTRATE_TEMP, CRITICAL_TEMP, x, xint)
        column, row = convert_cell_to_coordinates(cell)
        ax.plot(
            x, temp, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Enable Write Current ($\mu$A)")
    ax.set_ylabel("Channel Temperature (K)")
    ax.set_ylim(0, 13)
    ax.hlines(CRITICAL_TEMP, 0, 500, linestyle="--")
    return ax


def plot_enable_read_temperature():
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for cell in CELLS.keys():
        xint = CELLS[cell].get("x_intercept")
        x = CELLS[cell].get("enable_read_current") * 1e6
        temp = calculate_channel_temperature(1.3, 12.3, x, xint)
        column, row = convert_cell_to_coordinates(cell)
        ax.plot(
            x, temp, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Enable Read Current ($\mu$A)")
    ax.set_ylabel("Channel Temperature (K)")
    ax.set_ylim(0, 13)
    ax.hlines(CRITICAL_TEMP, 0, 500, linestyle="--")


def plot_write_sweep(ax: Axes, dict_list: str) -> Axes:
    colors = CMAP(np.linspace(0, 1, len(dict_list)))
    ax2 = ax.twinx()
    for data_dict in dict_list:
        x, y, ztotal = build_array(data_dict, "bit_error_rate")
        _, _, zswitch = build_array(data_dict, "total_switches_norm")
        # write_temp = get_write_temperature(data_dict)
        ax.plot(
            y,
            ztotal,
            # label=f"$T_{{W}}$ = {write_temp:.2f} K",
            color=colors[dict_list.index(data_dict)],
        )
        ax2.plot(
            y,
            zswitch,
            label="_",
            color="grey",
            linewidth=0.5,
            linestyle=":",
        )

        critical_current = calculate_critical_current(data_dict)

    ax.legend(frameon=False, bbox_to_anchor=(1.1, 1), loc="upper left")
    ax.set_ylim([0, 1])
    ax.set_xlabel("Write Current ($\mu$A)")
    ax.set_ylabel("Bit Error Rate")
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("Switching Probability")
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    return ax


if __name__ == "__main__":
    dict_list = import_directory(
        r"C:\Users\ICE\Documents\GitHub\nmem\src\nmem\analysis\enable_write_current_sweep\data"
    )

    fig, ax = plt.subplots()
    plot_enable_write_sweep_multiple(ax, dict_list[::2])
    plt.show()
