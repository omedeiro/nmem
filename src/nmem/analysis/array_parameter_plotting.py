import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    convert_location_to_coordinates,
    plot_array,
)
from nmem.calculations.calculations import (
    calculate_heater_power,
    htron_critical_current,
)
from nmem.measurement.cells import CELLS

# plt.rcParams["figure.figsize"] = [10, 12]
plt.rcParams["font.size"] = 10


def initialize_dict(array_size: tuple) -> dict:
    return {
        "write_current": np.zeros(array_size),
        "write_current_norm": np.zeros(array_size),
        "read_current": np.zeros(array_size),
        "read_current_norm": np.zeros(array_size),
        "slope": np.zeros(array_size),
        "intercept": np.zeros(array_size),
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
    param_dict["intercept"][y, x] = cell["intercept"]
    param_dict["resistance"][y, x] = cell["resistance_cryo"]
    param_dict["bit_error_rate"][y, x] = cell.get("min_bit_error_rate", np.nan)
    param_dict["max_critical_current"][y, x] = cell.get("max_critical_current", np.nan) * 1e6
    if cell["intercept"] != 0:
        write_critical_current = htron_critical_current(
            cell["enable_write_current"] * 1e6, cell["slope"], cell["intercept"]
        )
        read_critical_current = htron_critical_current(
            cell["enable_read_current"] * 1e6, cell["slope"], cell["intercept"]
        )
        param_dict["x_intercept"][y, x] = -cell["intercept"] / cell["slope"]
        param_dict["write_current_norm"][y, x] = cell["write_current"] * 1e6 / write_critical_current
        param_dict["read_current_norm"][y, x] = cell["read_current"] * 1e6 / read_critical_current
        param_dict["enable_write_power"][y, x] = calculate_heater_power(
            cell["enable_write_current"] * 1e-6, cell["resistance_cryo"]
        )
        param_dict["enable_write_current_norm"][y, x] = cell["enable_write_current"] * 1e6 / param_dict["x_intercept"][y, x]
        param_dict["enable_read_power"][y, x] = calculate_heater_power(
            cell["enable_read_current"] * 1e-6, cell["resistance_cryo"]
        )
        param_dict["enable_read_current_norm"][y, x] = cell["enable_read_current"] * 1e6 / param_dict["x_intercept"][y, x]
    return param_dict


if __name__ == "__main__":
    xloc_list = []
    yloc_list = []
    ARRAY_SIZE = (4, 4)
    param_dict = initialize_dict(ARRAY_SIZE)
    for c in CELLS:
        xloc, yloc = convert_location_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)
    
    fig, ax = plt.subplots()
    plot_array(ax, xloc_list, yloc_list, param_dict["write_current"], "Write Current [$\mu$A]")
  