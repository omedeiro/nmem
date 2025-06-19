from typing import Literal, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import least_squares

from nmem.analysis.bit_error import (
    get_bit_error_rate,
    get_bit_error_rate_args,
)
from nmem.analysis.constants import (
    CRITICAL_CURRENT_ZERO,
    CRITICAL_TEMP,
    IRHL_TR,
    IRM,
    SUBSTRATE_TEMP,
    WIDTH,
)
from nmem.analysis.currents import (
    calculate_channel_temperature,
    calculate_critical_current_temp,
    calculate_state_currents,
    get_channel_temperature,
    get_channel_temperature_sweep,
    get_critical_current_heater_off,
    get_enable_write_current,
    get_read_current,
    get_read_currents,
    get_write_current,
)
from nmem.analysis.utils import (
    convert_cell_to_coordinates,
    filter_first,
)
from nmem.calculations.calculations import (
    calculate_heater_power,
    htron_critical_current,
)
from nmem.measurement.cells import CELLS


def calculate_inductance_ratio(state0, state1, ic0):
    alpha = (ic0 - state1) / (state0 - state1)
    # alpha_test = 1 - ((critical_current_right - persistent_current_est) / ic)
    # alpha_test2 = (critical_current_left - persistent_current_est) / ic2

    return alpha


def calculate_operating_table(
    dict_list, ic_list, write_current_list, ic_list2, write_current_list2
):
    ic = np.array(ic_list)
    ic2 = np.array(ic_list2)
    write_current_array = np.array(write_current_list)
    data_dict = dict_list[-1] if dict_list else {}
    read_temperature = calculate_channel_temperature(
        CRITICAL_TEMP,
        SUBSTRATE_TEMP,
        data_dict["enable_read_current"] * 1e6,
        CELLS[data_dict["cell"][0]]["x_intercept"],
    ).flatten()
    write_temperature = calculate_channel_temperature(
        CRITICAL_TEMP,
        SUBSTRATE_TEMP,
        data_dict["enable_write_current"] * 1e6,
        CELLS[data_dict["cell"][0]]["x_intercept"],
    ).flatten()
    critical_current_channel = calculate_critical_current_temp(
        read_temperature, CRITICAL_TEMP, CRITICAL_CURRENT_ZERO
    )
    critical_current_left = critical_current_channel * WIDTH
    critical_current_right = critical_current_channel * (1 - WIDTH)
    read_current_difference = ic2 - ic
    read1_dist = np.abs(ic - IRM)
    read2_dist = np.abs(ic2 - IRM)
    persistent_current = []
    for i, write_current in enumerate(write_current_list):
        if write_current > IRHL_TR / 2:
            ip = np.abs(write_current - IRHL_TR) / 2
        else:
            ip = write_current
        if ip > IRHL_TR / 2:
            ip = IRHL_TR / 2
        persistent_current.append(ip)
    df = pd.DataFrame(
        {
            "Write Current": write_current_list,
            "Read Current": ic_list,
            "Read Current 2": ic_list2,
            "Channel Critical Current": critical_current_channel
            * np.ones_like(ic_list),
            "Channel Critical Current Left": critical_current_left
            * np.ones_like(ic_list),
            "Channel Critical Current Right": critical_current_right
            * np.ones_like(ic_list),
            "Persistent Current": persistent_current,
            "Read Current Difference": read_current_difference,
            "Read Current 1 Distance": read1_dist,
            "Read Current 2 Distance": read2_dist,
        }
    )
    return df


def get_enable_write_width(data_dict: dict) -> float:
    return filter_first(data_dict.get("enable_write_width"))


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


def get_fitting_points(
    x: np.ndarray, y: np.ndarray, ztotal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mid_idx = np.where(ztotal > np.nanmax(ztotal, axis=0) / 2)
    xfit, xfit_idx = np.unique(x[mid_idx[1]], return_index=True)
    yfit = y[mid_idx[0]][xfit_idx]
    return xfit, yfit


def get_voltage_trace_data(
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

def process_ber_data(logger=None) -> np.ndarray:
    """
    Process bit error rate (BER) data and return key statistics.
    """
    param_dict = initialize_dict((4, 4))
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)

    ber_array = param_dict["bit_error_rate"]
    valid_ber = ber_array[np.isfinite(ber_array) & (ber_array < 5.5e-2)]

    average_ber = np.mean(valid_ber)
    std_ber = np.std(valid_ber)
    min_ber = np.min(valid_ber)
    max_ber = np.max(valid_ber)

    logger.info("=== Array BER Statistics ===")
    logger.info(f"Average BER: {average_ber:.2e}")
    logger.info(f"Std Dev BER: {std_ber:.2e}")
    logger.info(f"Min BER: {min_ber:.2e}")
    logger.info(f"Max BER: {max_ber:.2e}")
    logger.info("=============================")

    return ber_array


def analyze_alignment_stats(df_z, df_rot_valid, dx_nm, dy_nm):
    """
    Compute statistics for z height and rotation.
    Returns: z_mean, z_std, r_mean, r_std
    """
    z_mean, z_std = df_z["z_height_mm"].mean(), df_z["z_height_mm"].std()
    r_mean, r_std = (
        df_rot_valid["rotation_mrad"].mean(),
        df_rot_valid["rotation_mrad"].std(),
    )
    return z_mean, z_std, r_mean, r_std


def analyze_geom_loop_size(data, loop_sizes, nmeas=1000):
    """
    Analyzes geometry loop size data.
    Returns:
        vch_list: list of Vch arrays
        ber_est_list: list of BER arrays
        err_list: list of error arrays
        best_ber: list of minimum BER for each loop size
    """
    import numpy as np

    vch_list = []
    ber_est_list = []
    err_list = []
    best_ber = []
    for i, d in enumerate(data):
        Vch = np.ravel(d["Vch"]) * 1e3
        ber_est = np.ravel(d["ber_est"])
        err = np.sqrt(ber_est * (1 - ber_est) / nmeas)
        vch_list.append(Vch)
        ber_est_list.append(ber_est)
        err_list.append(err)
        best_ber.append(np.min(ber_est))
    return vch_list, ber_est_list, err_list, best_ber


def interpolate_map(data_map, radius, grid_x, grid_y, boundary_pts):
    x_vals = data_map.columns.astype(float)
    y_vals = data_map.index.astype(float)
    xy = np.array([(x, y) for y in y_vals for x in x_vals])
    z = data_map.values.flatten()
    mask = ~np.isnan(z)
    xy, z = xy[mask], z[mask]

    bx, by, bz = boundary_pts
    aug_xy = np.column_stack(
        [np.concatenate([xy[:, 0], bx]), np.concatenate([xy[:, 1], by])]
    )
    aug_z = np.concatenate([z, bz])

    grid_z = griddata(aug_xy, aug_z, (grid_x, grid_y), method="cubic")
    distance = np.sqrt(grid_x**2 + grid_y**2)
    grid_z[distance > radius] = np.nan
    return grid_z, xy, z


def analyze_prbs_errors(data_list, trim=4500):
    W1R0_error = 0
    W0R1_error = 0
    error_locs = []
    for i, data in enumerate(data_list):
        bit_write = "".join(data["bit_string"].flatten())
        bit_read = "".join(data["byte_meas"].flatten())
        errors = [bw != br for bw, br in zip(bit_write, bit_read)]
        for j, error in enumerate(errors):
            if error:
                if bit_write[j] == "1":
                    W1R0_error += 1
                elif bit_write[j] == "0":
                    W0R1_error += 1
                error_locs.append((i, j, bit_write[j], bit_read[j]))
    total_error = W1R0_error + W0R1_error
    return total_error, W1R0_error, W0R1_error, error_locs


def fit_state_currents(
    x_list,
    y_list,
    initial_guess,
    width,
    critical_current_zero,
    bounds=([0, -100], [1, 100]),
):

    def model_function(x0, x1, x2, x3, alpha, persistent):
        retrap = 1
        i0, _, _, _ = calculate_state_currents(
            x0, CRITICAL_TEMP, retrap, width, alpha, persistent, critical_current_zero
        )
        _, i1, _, _ = calculate_state_currents(
            x1, CRITICAL_TEMP, retrap, width, alpha, persistent, critical_current_zero
        )
        _, _, i2, _ = calculate_state_currents(
            x2, CRITICAL_TEMP, retrap, width, alpha, persistent, critical_current_zero
        )
        _, _, _, i3 = calculate_state_currents(
            x3, CRITICAL_TEMP, retrap, width, alpha, persistent, critical_current_zero
        )
        model = [i0, i1, i2, i3]
        return model

    def residuals(p, x0, y0, x1, y1, x2, y2, x3, y3):
        alpha, persistent = p
        model = model_function(x0, x1, x2, x3, alpha, persistent)
        residuals = np.concatenate(
            [
                y0 - model[0],
                y1 - model[1],
                y2 - model[2],
                y3 - model[3],
            ]
        )
        return residuals

    fit = least_squares(
        residuals,
        initial_guess,
        args=(
            x_list[0],
            y_list[0],
            x_list[1],
            y_list[1],
            x_list[2],
            y_list[2],
            x_list[3],
            y_list[3],
        ),
        bounds=bounds,
    )
    return fit


def prepare_state_current_data(data_dict):
    from nmem.analysis.utils import filter_nan

    temp = data_dict["measured_temperature"].flatten()
    state_currents = data_dict["measured_state_currents"]
    x_list = []
    y_list = []
    for i in range(4):
        x = temp
        y = state_currents[:, i]
        x, y = filter_nan(x, y)
        if len(x) > 0:
            x_list.append(x)
            y_list.append(y)
        else:
            x_list.append(None)
            y_list.append(None)
    return x_list, y_list


def compute_sigma_separation(data: dict, show_print=True) -> float:
    """Compute the peak separation between read0 and read1 histograms in units of σ."""
    v_read0 = np.array(data["read_zero_top"])
    v_read1 = np.array(data["read_one_top"])

    # Remove NaNs or invalid data
    v_read0 = v_read0[np.isfinite(v_read0)]
    v_read1 = v_read1[np.isfinite(v_read1)]

    mu0 = np.mean(v_read0)
    mu1 = np.mean(v_read1)
    sigma0 = np.std(v_read0)
    sigma1 = np.std(v_read1)

    sigma_avg = 0.5 * (sigma0 + sigma1)
    separation_sigma = mu0 + sigma0 * 3 - (mu1 - 3 * sigma1)

    if show_print:
        print(f"μ0 = {mu0:.3f} mV, σ0 = {sigma0:.3f} mV")
        print(f"μ1 = {mu1:.3f} mV, σ1 = {sigma1:.3f} mV")
        print(f"Separation = {separation_sigma:.2f} σ")

    return separation_sigma


def extract_shifted_traces(
    data_dict: dict, trace_index: int = 0, time_shift: float = 0.0
) -> Tuple:
    chan_in_x, chan_in_y = get_voltage_trace_data(
        data_dict, "trace_chan_in", trace_index
    )
    chan_out_x, chan_out_y = get_voltage_trace_data(
        data_dict, "trace_chan_out", trace_index
    )
    enab_in_x, enab_in_y = get_voltage_trace_data(data_dict, "trace_enab", trace_index)

    # Shift all x values
    chan_in_x = chan_in_x + time_shift
    chan_out_x = chan_out_x + time_shift
    enab_in_x = enab_in_x + time_shift

    return chan_in_x, chan_in_y, enab_in_x, enab_in_y, chan_out_x, chan_out_y


def extract_temp_current_data(dict_list):
    data = []
    data2 = []
    for data_dict in dict_list:
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_currents = get_read_currents(data_dict)
        enable_write_current = get_enable_write_current(data_dict)
        read_current = get_read_current(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                entry = {
                    "write_current": write_currents[arg],
                    "write_temp": get_channel_temperature(data_dict, "write"),
                    "read_current": read_current,
                    "enable_write_current": enable_write_current,
                }
                if i == 0:
                    data.append(entry)
                if i == 2:
                    data2.append(entry)
    return data, data2


def process_write_temp_arrays(dict_list):
    write_temp_array = np.empty((len(dict_list), 4))
    write_current_array = np.empty((len(dict_list), 1))
    critical_current_zero = None
    for j, data_dict in enumerate(dict_list):
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_current = get_write_current(data_dict)
        write_temps = get_channel_temperature_sweep(data_dict)
        write_current_array[j] = write_current
        critical_current_zero = get_critical_current_heater_off(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                write_temp_array[j, i] = write_temps[arg]
    return write_current_array, write_temp_array, critical_current_zero


def process_array_parameter_data(cells, array_size=(4, 4)):
    """
    Process cell data for parameter arrays for the given array size.
    Returns xloc_list, yloc_list, param_dict, yintercept_list, slope_list.
    """
    from nmem.analysis.core_analysis import initialize_dict, process_cell
    from nmem.analysis.utils import convert_cell_to_coordinates

    xloc_list = []
    yloc_list = []
    param_dict = initialize_dict(array_size)
    yintercept_list = []
    slope_list = []
    for c in cells:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(cells[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)
        yintercept = cells[c]["y_intercept"]
        yintercept_list.append(yintercept)
        slope = cells[c]["slope"]
        slope_list.append(slope)
    return xloc_list, yloc_list, param_dict, yintercept_list, slope_list
