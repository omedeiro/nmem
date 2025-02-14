import matplotlib.pyplot as plt


from nmem.measurement.cells import CELLS
from nmem.analysis.analysis import calculate_critical_current_zero
import pandas as pd

if __name__ == "__main__":
    cell_list = []
    critical_current_list = []
    critical_current_zero_list = []
    critical_current_intercept_list = []
    max_heater_current_list = []
    enable_read_current_list = []
    enable_write_current_list = []
    resistance_cryo_list = []
    fig, ax = plt.subplots()
    for c in CELLS:
        cell_list.append(c)
        max_critical_current = CELLS[c]["max_critical_current"] * 1e6
        critical_current_list.append(max_critical_current)

        critical_current_zero = calculate_critical_current_zero(
            12.3, 1.3, max_critical_current
        )
        critical_current_zero_list.append(critical_current_zero)
        critical_current_intercept = CELLS[c]["y_intercept"]
        critical_current_intercept_list.append(critical_current_intercept)
        max_heater_current = CELLS[c]["x_intercept"]
        max_heater_current_list.append(max_heater_current)
        enable_read_current = CELLS[c]["enable_read_current"] * 1e6
        enable_read_current_list.append(enable_read_current)
        enable_write_current = CELLS[c]["enable_write_current"] * 1e6
        enable_write_current_list.append(enable_write_current)
        resistance_cryo = CELLS[c]["resistance_cryo"]
        resistance_cryo_list.append(resistance_cryo)

    df = pd.DataFrame(
        {
            "Cell": cell_list,
            "Critical Current": critical_current_list,
            "Zero": critical_current_zero_list,
            "Y-Intercept": critical_current_intercept_list,
            "Max Heater Current": max_heater_current_list,
            "Enable Read Current": enable_read_current_list,
            "Enable Write Current": enable_write_current_list,
            "Resistance Cryo": resistance_cryo_list,
        },
        columns=[
            "Cell",
            "Critical Current",
            "Zero",
            "Y-Intercept",
            "Max Heater Current",
            "Enable Read Current",
            "Enable Write Current",
            "Resistance Cryo",
        ],
    )
    ax.plot(df["Cell"], df["Y-Intercept"], "o")
    print(df)
