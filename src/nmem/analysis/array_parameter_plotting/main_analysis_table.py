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
    df = pd.DataFrame(
        {
            "Cell": cell_list,
            "Critical Current": critical_current_list,
            "Zero": critical_current_zero_list,
            "Y-Intercept": critical_current_intercept_list,
            "Max Heater Current": max_heater_current_list,
        },
        columns=[
            "Cell",
            "Critical Current",
            "Zero",
            "Y-Intercept",
            "Max Heater Current",
        ],
    )

    print(df)
