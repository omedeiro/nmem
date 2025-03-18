import matplotlib.pyplot as plt
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
if __name__ == "__main__":

    fig, ax = plt.subplots()
    data = []
    cell_dictionary = CELLS
    for c in cell_dictionary:
        critical_current_tsub = cell_dictionary[c]["max_critical_current"] * 1e6

        critical_current_tzero = calculate_critical_current_zero(
            CRITICAL_TEMP, SUBSTRATE_TEMP, critical_current_tsub
        )
        critical_current_intercept = cell_dictionary[c]["y_intercept"]
        critical_current_tzero_alpha = critical_current_intercept * ALPHA_FIT

        max_heater_current = cell_dictionary[c]["x_intercept"]
        enable_read_current = cell_dictionary[c]["enable_read_current"] * 1e6
        enable_write_current = cell_dictionary[c]["enable_write_current"] * 1e6
        resistance_cryo = cell_dictionary[c]["resistance_cryo"]

        read_temperature = calculate_channel_temperature(
            CRITICAL_TEMP, SUBSTRATE_TEMP, enable_read_current, max_heater_current
        ).flatten()[0]
        write_temperature = calculate_channel_temperature(
            CRITICAL_TEMP, SUBSTRATE_TEMP, enable_write_current, max_heater_current
        ).flatten()[0]

        critical_current_density_zero = (
            critical_current_tsub
            * 1e-6
            / (WIDTH_LEFT * THICKNESS + WIDTH_RIGHT * THICKNESS)
        ) * (1 - (SUBSTRATE_TEMP / CRITICAL_TEMP) ** 3) ** -2.1


        data.append(
            {
                "Cell": c,
                "Critical Current Tsub": critical_current_tsub,
                # "Critical Current Tzero": critical_current_tzero,
                "Y-Intercept": critical_current_intercept,
                # "Critical Current Tzero Alpha": critical_current_tzero_alpha,
                "Max Heater Current": max_heater_current,
                "Read Temperature": read_temperature,
                "Write Temperature": write_temperature,
                # "Enable Read Current": enable_read_current,
                # "Enable Write Current": enable_write_current,
                "Resistance Cryo": resistance_cryo,
                "Critical Current Density Zero [MA/cm2]": critical_current_density_zero
                * 1e-10,
            }
        )

    df = pd.DataFrame(data)
    df.style.format("{:.3f}")

    df.to_csv("array_parameter_table.csv", float_format="%.3f")
