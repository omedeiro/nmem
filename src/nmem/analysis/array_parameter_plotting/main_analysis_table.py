

import pandas as pd

from nmem.analysis.analysis import (
    CRITICAL_TEMP,
    SUBSTRATE_TEMP,
    calculate_channel_temperature,
    calculate_critical_current_zero,
)
from nmem.measurement.cells import CELLS

ALPHA_FIT = 0.88
BETA_FIT = 1.25
WIDTH_LEFT = 100e-9
WIDTH_RIGHT = 300e-9
THICKNESS = 23e-9


def process_cell(cell_name: str, cell_dict: dict):
    # Import
    cell_data = cell_dict[cell_name]
    critical_current_tsub = cell_data["max_critical_current"] * 1e6
    critical_current_intercept = cell_data["y_intercept"]
    max_heater_current = cell_data["x_intercept"]
    enable_read_current = cell_data["enable_read_current"] * 1e6
    enable_write_current = cell_data["enable_write_current"] * 1e6
    resistance_cryo = cell_data["resistance_cryo"]
    x_intercept = cell_data["x_intercept"]

    # Preprocess
    critical_current_tzero_alpha = critical_current_intercept * ALPHA_FIT
    read_temperature = calculate_channel_temperature(
        CRITICAL_TEMP, SUBSTRATE_TEMP, enable_read_current, max_heater_current
    ).flatten()[0]
    write_temperature = calculate_channel_temperature(
        CRITICAL_TEMP, SUBSTRATE_TEMP, enable_write_current, max_heater_current
    ).flatten()[0]

    critical_current_tzero = calculate_critical_current_zero(
        CRITICAL_TEMP, SUBSTRATE_TEMP, critical_current_tsub
    )
    critical_current_density_zero = (
        critical_current_tsub
        * 1e-6
        / (WIDTH_LEFT * THICKNESS + WIDTH_RIGHT * THICKNESS)
    ) * (1 - (SUBSTRATE_TEMP / CRITICAL_TEMP) ** 3) ** -2.1

    return {
        "Cell Name": cell_name,
        "Critical Current Tsub": critical_current_tsub,
        "Critical Current Intercept": critical_current_intercept,
        "Max Heater Current": max_heater_current,
        "Enable Read Current": enable_read_current,
        "Enable Write Current": enable_write_current,
        "Resistance Cryo": resistance_cryo,
        "X Intercept": x_intercept,
        "Critical Current Tzero Alpha": critical_current_tzero_alpha,
        "Read Temperature": read_temperature,
        "Write Temperature": write_temperature,
        "Critical Current Tzero": critical_current_tzero,
        "Critical Current Density Zero (MA/cm2)": critical_current_density_zero * 1e-10,
    }


if __name__ == "__main__":
    cell_dictionary = CELLS
    cell_data_list = [
        process_cell(cell_name, cell_dictionary) for cell_name in cell_dictionary.keys()
    ]
    df = pd.DataFrame(cell_data_list)

    df.to_csv("array_parameter_table.csv", float_format="%.3f")
