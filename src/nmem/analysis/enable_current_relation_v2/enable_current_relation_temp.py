import matplotlib.pyplot as plt

from nmem.analysis.constants import CRITICAL_TEMP, SUBSTRATE_TEMP
from nmem.analysis.core_analysis import (
    get_fitting_points,
)
from nmem.analysis.currents import (
    calculate_channel_temperature,
    get_max_enable_current,
)
from nmem.analysis.data_import import import_directory
from nmem.analysis.sweep_plots import plot_enable_current_vs_temp
from nmem.analysis.utils import (
    convert_cell_to_coordinates,
    filter_plateau,
    get_current_cell,
)


def process_enable_current_data(dict_list):
    """
    Extracts and returns all arrays needed for plotting from the data list.
    Returns a list of dicts with cell, column, row, enable_currents, switching_current, ztotal, xfit, yfit, max_enable_current, channel_temperature.
    """
    processed = []
    for data_dict in dict_list:
        cell = get_current_cell(data_dict)
        column, row = convert_cell_to_coordinates(cell)
        enable_currents = data_dict["x"][0]
        switching_current = data_dict["y"][0]
        ztotal = data_dict["ztotal"]
        xfit, yfit = get_fitting_points(enable_currents, switching_current, ztotal)
        xfit_plateau, yfit_plateau = filter_plateau(xfit, yfit, yfit[0] * 0.75)
        max_enable_current = get_max_enable_current(data_dict)
        channel_temperature = calculate_channel_temperature(
            CRITICAL_TEMP, SUBSTRATE_TEMP, enable_currents, max_enable_current
        )
        processed.append(
            {
                "cell": cell,
                "column": column,
                "row": row,
                "enable_currents": enable_currents,
                "switching_current": switching_current,
                "ztotal": ztotal,
                "xfit": xfit,
                "yfit": yfit,
                "xfit_plateau": xfit_plateau,
                "yfit_plateau": yfit_plateau,
                "max_enable_current": max_enable_current,
                "channel_temperature": channel_temperature,
            }
        )
    return processed

def main(data_dir="data", save_fig=False, output_path="enable_current_vs_temp.png"):
    """
    Main function to process data and plot enable current vs. temperature.
    """
    dict_list = import_directory(data_dir)
    data = process_enable_current_data(dict_list)
    fig, axs, axs2 = plot_enable_current_vs_temp(data, save_fig, output_path)
    plt.show()


if __name__ == "__main__":
    main()
