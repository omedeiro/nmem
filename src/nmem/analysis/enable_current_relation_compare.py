import matplotlib.pyplot as plt
import numpy as np
from nmem.calculations.calculations import htron_critical_current, htron_heater_current
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


def plot_text_labels(xloc, yloc, ztotal):
    for x, y in zip(xloc, yloc):
        plt.text(
            x,
            y,
            f"{ztotal[y, x]:.2f}",
            fontsize=12,
            color="black",
            backgroundcolor="none",
            ha="center",
            va="center",
            weight="bold",
        )


def plot_array(xloc, yloc, ztotal, title):
    fig, ax = plt.subplots()
    im = ax.imshow(ztotal, cmap="viridis")
    plt.title(title)
    plt.xticks(range(4), ["A", "B", "C", "D"])
    plt.yticks(range(4), ["1", "2", "3", "4"])
    plt.xlabel("Column")
    plt.ylabel("Row")
    cbar = plt.colorbar(im)
    plot_text_labels(xloc, yloc, ztotal)
    plt.show()


if __name__ == "__main__":
    xloc = []
    yloc = []
    slope_array = np.zeros((4, 4))
    intercept_array = np.zeros((4, 4))
    x_intercept_array = np.zeros((4, 4))
    write_array = np.zeros((4, 4))
    read_array = np.zeros((4, 4))
    resistance_array = np.zeros((4, 4))
    bit_error_array = np.zeros((4, 4))
    max_critical_current_array = np.zeros((4, 4))
    for c in CELLS:
        write_current = CELLS[c]["write_current"] * 1e6
        read_current = CELLS[c]["read_current"] * 1e6
        enable_write_current = CELLS[c]["enable_write_current"] * 1e6
        enable_read_current = CELLS[c]["enable_read_current"] * 1e6
        slope = CELLS[c]["slope"]
        intercept = CELLS[c]["intercept"]
        resistance = CELLS[c]["resistance_cryo"]
        bit_error_rate = CELLS[c].get("min_bit_error_rate", 0)
        max_critical_current = CELLS[c].get("max_critical_current", 0) * 1e6
        if intercept != 0:
            write_critical_current = htron_critical_current(
                enable_write_current, slope, intercept
            )
            read_critical_current = htron_critical_current(
                enable_read_current, slope, intercept
            )
            write_heater_current = htron_heater_current(write_current, slope, intercept)
            read_heater_current = htron_heater_current(read_current, slope, intercept)
            x, y = convert_location_to_coordinates(c)
            xloc.append(x)
            yloc.append(y)
            slope_array[y, x] = slope
            intercept_array[y, x] = intercept
            max_heater_current = -intercept / slope
            x_intercept_array[y, x] = max_heater_current
            write_array[y, x] = write_current / write_critical_current
            read_array[y, x] = read_current / read_critical_current
            resistance_array[y, x] = resistance
            bit_error_array[y, x] = bit_error_rate
            max_critical_current_array[y, x] = max_critical_current
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
    plot_array(xloc, yloc, write_array, "Write Current Normalized")
    plot_array(xloc, yloc, read_array, "Read Current Normalized")
    plot_array(xloc, yloc, np.abs(slope_array), "Slope [$\mu$A/$\mu$A]")
    plot_array(xloc, yloc, intercept_array, "Y-Intercept [$\mu$A]")
    # plot_array(xloc, yloc, resistance_array, "Resistance [$\Omega$]")
    plot_array(xloc, yloc, x_intercept_array, "X-Intercept [$\mu$A]")
    plot_array(xloc, yloc, bit_error_array, "Bit Error Rate")
    plot_array(xloc, yloc, max_critical_current_array, "Max Critical Current [$\mu$A]")
