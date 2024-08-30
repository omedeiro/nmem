import matplotlib.pyplot as plt
import numpy as np

from nmem.calculations.calculations import (
    calculate_heater_power,
    htron_critical_current,
    htron_heater_current,
)
from nmem.measurement.cells import CELLS


def create_color_map(min_slope, max_slope):
    """Creates a colormap from min_slope to max_slope."""
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "my_cmap", [(0, "blue"), (0.5, "white"), (1, "red")]
    )
    norm = mcolors.Normalize(vmin=min_slope, vmax=max_slope)
    return cmap, norm


def convert_location_to_coordinates(location):
    """Converts a location like 'A1' to coordinates (x, y)."""
    column_letter = location[0]
    row_number = int(location[1:]) - 1
    column_number = ord(column_letter) - ord("A")
    return column_number, row_number


def plot_text_labels(xloc, yloc, ztotal, log=False):
    for x, y in zip(xloc, yloc):
        text = f"{ztotal[y, x]:.2f}"
        if log:
            text = f"{ztotal[y, x]:.1e}"
        txt_color = "black"
        if ztotal[y, x] < 0.5 * max(ztotal.flatten()):
            txt_color = "white"

        plt.text(
            x,
            y,
            text,
            fontsize=12,
            color=txt_color,
            backgroundcolor="none",
            ha="center",
            va="center",
            weight="bold",
        )


def plot_array(xloc, yloc, ztotal, title, log=False, norm=False, reverse=False):
    fig, ax = plt.subplots()

    cmap = plt.cm.get_cmap("viridis")
    if reverse:
        cmap = plt.cm.get_cmap("viridis").reversed()

    if norm:
        im = ax.imshow(ztotal, cmap=cmap, vmin=0, vmax=1)
    else:
        im = ax.imshow(ztotal, cmap=cmap)
    plt.title(title)
    plt.xticks(range(4), ["A", "B", "C", "D"])
    plt.yticks(range(4), ["1", "2", "3", "4"])
    plt.xlabel("Column")
    plt.ylabel("Row")
    cbar = plt.colorbar(im)

    plot_text_labels(xloc, yloc, ztotal, log)

    plt.show()


def plot_normalization(
    write_current_norm: np.ndarray,
    read_current_norm: np.ndarray,
    enable_write_current: np.ndarray,
    enable_read_current: np.ndarray,
):
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

    fig, ax = plt.subplots()
    ax.boxplot(write_current_norm.flatten(), positions=[0], widths=0.5)
    ax.boxplot(read_current_norm.flatten(), positions=[1], widths=0.5)
    ax.boxplot(enable_write_current.flatten(), positions=[2], widths=0.5)
    ax.boxplot(enable_read_current.flatten(), positions=[3], widths=0.5)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(
        ["Write Current", "Read Current", "Enable\nWrite Current", "Enable\nRead Current"]
    )
    ax.set_xlabel("Input Type")
    plt.xticks(rotation=45)
    ax.set_ylabel("Normalized Current")
    ax.set_yticks(np.linspace(0, 1, 11))

    plt.show()


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
            print(f"Cell: {c}")
            print(f"Write Current: {write_current:.2f}")
            print(f"Write Critical Current: {write_critical_current:.2f}")
            print(
                f"Write Current Normalized: {write_current/write_critical_current:.2f}"
            )
            print(f"Read Current: {read_current:.2f}")
            print(f"Read Critical Current: {read_critical_current:.2f}")
            print(f"Read Current Normalized: {read_current/read_critical_current:.2f}")
            print(f"Write Heater Current: {write_heater_current}")
            print(
                f"Write Heater Current Normalized: {write_heater_current/max_heater_current}"
            )
            print(f"Read Heater Current: {read_heater_current}")
            print(
                f"Read Heater Current Normalized: {read_heater_current/max_heater_current}"
            )
            print("\n")

    ztotal = write_array
    plot_array(xloc, yloc, ztotal, "Write Current [$\mu$A]")
    plot_array(xloc, yloc, write_array_norm, "Write Current Normalized")
    plot_array(xloc, yloc, read_array, "Read Current [$\mu$A]")
    plot_array(xloc, yloc, read_array_norm, "Read Current Normalized")
    # plot_array(xloc, yloc, np.abs(slope_array), "Slope [$\mu$A/$\mu$A]")
    # plot_array(xloc, yloc, intercept_array, "Y-Intercept [$\mu$A]")
    # plot_array(xloc, yloc, x_intercept_array, "X-Intercept [$\mu$A]")
    plot_array(xloc, yloc, bit_error_array, "Bit Error Rate", log=True)
    # plot_array(xloc, yloc, max_critical_current_array, "Max Critical Current [$\mu$A]")
    plot_array(xloc, yloc, enable_write_array, "Enable Write Current [$\mu$A]")
    plot_array(xloc, yloc, enable_read_array, "Enable Read Current [$\mu$A]")
    plot_array(
        xloc,
        yloc,
        enable_write_array / x_intercept_array,
        "Enable Write Current Normalized",
    )
    plot_array(
        xloc,
        yloc,
        enable_read_array / x_intercept_array,
        "Enable Read Current Normalized",
    )
    # plot_array(xloc, yloc, enable_write_power_array * 1e6, "Enable Write Power [uW]")
    # plot_array(xloc, yloc, enable_read_power_array * 1e6, "Enable Read Power [uW]")

    write_current_avg = np.mean(write_array[write_array > 0])
    write_current_std = np.std(write_array[write_array > 0])
    write_current_range = np.max(write_array[write_array > 0]) - np.min(
        write_array[write_array > 0]
    )
    write_current_min = np.min(write_array[write_array > 0])
    write_current_max = np.max(write_array[write_array > 0])
    read_current_avg = np.mean(read_array[read_array > 0])
    read_current_std = np.std(read_array[read_array > 0])
    read_current_range = np.max(read_array[read_array > 0]) - np.min(
        read_array[read_array > 0]
    )
    read_current_min = np.min(read_array[read_array > 0])
    read_current_max = np.max(read_array[read_array > 0])

    enable_write_avg = np.mean(enable_write_array[enable_write_array > 0])
    enable_write_std = np.std(enable_write_array[enable_write_array > 0])
    enable_write_min = np.min(enable_write_array[enable_write_array > 0])
    enable_write_max = np.max(enable_write_array[enable_write_array > 0])
    enable_read_avg = np.mean(enable_read_array[enable_read_array > 0])
    enable_read_std = np.std(enable_read_array[enable_read_array > 0])
    enable_read_min = np.min(enable_read_array[enable_read_array > 0])
    enable_read_max = np.max(enable_read_array[enable_read_array > 0])

    # print(f"Write Current Average [uA]: {write_current_avg:.2f}")
    # print(f"Write Current Std [uA]: {write_current_std:.2f}")
    # print(f"Write Current Range [uA]: {write_current_range:.2f}")

    # print(f"Read Current Average [uA]: {read_current_avg:.2f}")
    # print(f"Read Current Std [uA]: {read_current_std:.2f}")
    # print(f"Read Current Range [uA]: {read_current_range:.2f}")

    # print(
    #     f"Write Current Min, Max [uA]: {write_current_min:.2f}, {write_current_max:.2f}"
    # )
    # print(f"Read Current Min, Max [uA]: {read_current_min:.2f}, {read_current_max:.2f}")
    # print(f"Enable Write Average [uA]: {enable_write_avg:.2f}")
    # print(f"Enable Write Std [uA]: {enable_write_std:.2f}")
    # print(f"Enable Write Min, Max [uA]: {enable_write_min:.2f}, {enable_write_max:.2f}")

    # print(f"Enable Read Average [uA]: {enable_read_avg:.2f}")
    # print(f"Enable Read Std [uA]: {enable_read_std:.2f}")
    # print(f"Enable Read Min, Max [uA]: {enable_read_min:.2f}, {enable_read_max:.2f}")
    enable_write_array_norm = enable_write_array / x_intercept_array
    enable_read_array_norm = enable_read_array / x_intercept_array
    print(f"Average bit error rate {np.mean(bit_error_array):.2e}")
    plot_normalization(
        write_array_norm,
        read_array_norm,
        enable_write_array_norm,
        enable_read_array_norm,
    )