import matplotlib.pyplot as plt
import numpy as np

from nmem.analysis.analysis import (
    convert_location_to_coordinates,
    plot_array,
    plot_normalization,
)
from nmem.calculations.calculations import (
    calculate_heater_power,
    htron_critical_current,
    htron_heater_current,
)
from nmem.measurement.cells import CELLS
from matplotlib import font_manager

font_path = "/home/omedeiro/Inter-Regular.otf"  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["figure.figsize"] = [7, 3.5]
plt.rcParams["font.size"] = 5
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.family"] = "Inter"
plt.rcParams["lines.markersize"] = 2
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.labelpad"] = 0.5

if __name__ == "__main__":
    xloc = []
    yloc = []
    slope_array = np.zeros((4, 4))
    intercept_array = np.zeros((4, 4))
    x_intercept_array = np.zeros((4, 4))
    write_array = np.zeros((4, 4))
    write_array_norm = np.zeros((4, 4))
    read_array = np.zeros((4, 4))
    read_array_norm = np.zeros((4, 4))
    resistance_array = np.zeros((4, 4))
    bit_error_array = np.empty((4, 4))
    max_critical_current_array = np.zeros((4, 4))
    max_heater_current = np.zeros((4, 4))
    enable_write_array = np.zeros((4, 4))
    enable_read_array = np.zeros((4, 4))
    enable_write_power_array = np.zeros((4, 4))
    enable_read_power_array = np.zeros((4, 4))
    for c in CELLS:
        write_current = CELLS[c]["write_current"] * 1e6
        read_current = CELLS[c]["read_current"] * 1e6
        enable_write_current = CELLS[c]["enable_write_current"] * 1e6
        enable_read_current = CELLS[c]["enable_read_current"] * 1e6
        slope = CELLS[c]["slope"]
        intercept = CELLS[c]["intercept"]
        resistance = CELLS[c]["resistance_cryo"]
        bit_error_rate = CELLS[c].get("min_bit_error_rate", np.nan)
        max_critical_current = CELLS[c].get("max_critical_current", np.nan) * 1e6
        if intercept != 0:
            write_critical_current = htron_critical_current(
                enable_write_current, slope, intercept
            )
            read_critical_current = htron_critical_current(
                enable_read_current, slope, intercept
            )
            write_heater_current = htron_heater_current(write_current, slope, intercept)
            read_heater_current = htron_heater_current(read_current, slope, intercept)
            enable_write_power = calculate_heater_power(
                enable_write_current * 1e-6, resistance
            )
            enable_read_power = calculate_heater_power(
                enable_read_current * 1e-6, resistance
            )
            x, y = convert_location_to_coordinates(c)
            xloc.append(x)
            yloc.append(y)
            slope_array[y, x] = slope
            intercept_array[y, x] = intercept
            max_heater_current = -intercept / slope
            x_intercept_array[y, x] = max_heater_current
            write_array[y, x] = write_current
            write_array_norm[y, x] = write_current / write_critical_current
            read_array[y, x] = read_current
            read_array_norm[y, x] = read_current / read_critical_current
            resistance_array[y, x] = resistance
            bit_error_array[y, x] = bit_error_rate
            max_critical_current_array[y, x] = max_critical_current
            enable_write_array[y, x] = enable_write_current
            enable_read_array[y, x] = enable_read_current
            enable_write_power_array[y, x] = enable_write_power
            enable_read_power_array[y, x] = enable_read_power

    fig, axs = plt.subplot_mosaic(
        [
            ["bit_error", "max_heater_current", "delay"],
            ["max_critical_current", "read", "temperature"],
        ]
    )
    plot_array(xloc, yloc, bit_error_array, log=True, ax=axs["bit_error"])
    plot_array(
        xloc, yloc, max_critical_current_array, log=True, ax=axs["max_critical_current"]
    )
    plot_array(xloc, yloc, x_intercept_array, log=True, ax=axs["max_heater_current"])
    plot_array(xloc, yloc, read_array, log=True, ax=axs["read"])

    axs["delay"].set_xlabel("Delay ($\mu$s)")
    axs["delay"].set_ylabel("BER")

    axs["temperature"].set_xlabel("Temperature (K)")
    axs["temperature"].set_ylabel("BER")
    fig.patch.set_visible(False)
    plt.savefig("main_analysis.pdf", bbox_inches="tight")
