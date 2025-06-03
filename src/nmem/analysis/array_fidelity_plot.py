import os

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from nmem.analysis.analysis import (
    convert_cell_to_coordinates,
    get_bit_error_rate,
    get_bit_error_rate_args,
    get_channel_temperature,
    get_channel_temperature_sweep,
    get_critical_current_heater_off,
    get_enable_current_sweep,
    get_enable_write_current,
    get_read_current,
    get_read_currents,
    get_write_current,
    import_directory,
    initialize_dict,
    plot_enable_write_sweep_multiple,
    plot_parameter_array,
    plot_write_sweep,
    process_cell,
    set_inter_font,
    set_plot_style,
    set_pres_style,
)
from nmem.measurement.cells import CELLS

C0 = "#1b9e77"
C1 = "#d95f02"
RBCOLORS = plt.get_cmap("coolwarm")(np.linspace(0, 1, 4))
CMAP2 = plt.get_cmap("viridis")
set_pres_style()
set_inter_font()


# range set 1 [::2]
def plot_enable_sweep(
    ax: plt.Axes,
    dict_list: list[dict],
    range=None,
    add_errorbar=False,
    add_colorbar=False,
):
    N = len(dict_list)
    if range is not None:
        dict_list = dict_list[range]
    # ax, ax2 = plot_enable_write_sweep_multiple(ax, dict_list[0:6])
    ax = plot_enable_write_sweep_multiple(
        ax, dict_list, add_errorbar=add_errorbar, N=N, add_colorbar=add_colorbar
    )

    ax.set_ylabel("BER")
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    return ax


def plot_enable_sweep_markers(ax: plt.Axes, dict_list: list[dict]):
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.20))
    ax.set_ylim([8.3, 9.7])

    write_temp_array = np.empty((len(dict_list), 4))
    write_current_array = np.empty((len(dict_list), 1))
    enable_current_array = np.empty((len(dict_list), 4))
    for j, data_dict in enumerate(dict_list):
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_current = get_write_current(data_dict)
        write_temps = get_channel_temperature_sweep(data_dict)
        enable_currents = get_enable_current_sweep(data_dict)
        write_current_array[j] = write_current
        critical_current_zero = get_critical_current_heater_off(data_dict)
        for i, arg in enumerate(berargs):
            if arg is not np.nan:
                write_temp_array[j, i] = write_temps[arg]
                enable_current_array[j, i] = enable_currents[arg]
    markers = ["o", "s", "D", "^"]
    for i in range(4):
        ax.plot(
            enable_current_array[:, i],
            write_current_array,
            linestyle="--",
            marker=markers[i],
            markeredgecolor="k",
            markeredgewidth=0.5,
            color=RBCOLORS[i],
        )
    ax.set_ylim(0, 100)
    ax.set_xlim(250, 340)
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    ax.grid()
    ax.set_ylabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_xlabel("$I_{\mathrm{enable}}$ [$\mu$A]")
    ax.legend(
        [
            "$I_{1}$",
            "$I_{0}$",
            "$I_{0,\mathrm{inv}}$",
            "$I_{1,\mathrm{inv}}$",
        ],
        loc="lower left",
        frameon=True,
        ncol=1,
        facecolor="white",
        edgecolor="none",
    )


def plot_write_sweep_formatted(ax: plt.Axes, dict_list: list[dict]):
    plot_write_sweep(ax, dict_list)
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("BER")
    ax.set_xlim(0, 300)
    return ax


def plot_write_sweep_formatted_markers(ax: plt.Axes, data_dict: dict):
    data = data_dict.get("data")
    data2 = data_dict.get("data2")
    colors = CMAP2(np.linspace(0, 1, 4))
    ax.plot(
        [d["write_current"] for d in data],
        [d["write_temp"] for d in data],
        "d",
        color=colors[0],
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    ax.plot(
        [d["write_current"] for d in data2],
        [d["write_temp"] for d in data2],
        "o",
        color=colors[2],
        markeredgecolor="black",
        markeredgewidth=0.5,
    )
    ax.set_xlabel("$I_{\mathrm{write}}$ [$\mu$A]")
    ax.set_ylabel("$T_{\mathrm{write}}$ [K]")
    ax.set_xlim(0, 300)
    ax.legend(
        ["Lower bound", "Upper bound"],
        loc="upper right",
        fontsize=6,
        facecolor="white",
        frameon=True,
    )
    ax.grid()
    return ax


import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


def plot_delay(ax: plt.Axes, data_dict: dict):
    delay_list = np.array(data_dict.get("delay"))
    bit_error_rate = np.array(data_dict.get("bit_error_rate")).flatten()
    N = 200e3  # Total trials for BER standard deviation

    # Sort data for proper line plotting
    sort_index = np.argsort(delay_list)
    delay_list = delay_list[sort_index]
    bit_error_rate = bit_error_rate[sort_index]

    # Error bars calculation
    ber_std = np.sqrt(bit_error_rate * (1 - bit_error_rate) / N)

    # Plotting with improved style
    ax.errorbar(
        delay_list,
        bit_error_rate,
        yerr=ber_std,
        fmt="-o",
        color="black",
        elinewidth=1,
        capsize=2,
    )

    # Axes labels with specific font sizes
    ax.set_xlabel("$\Delta t$ [s]")
    ax.set_ylabel("BER")

    # Log scales with tick formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1e-6, delay_list.max() * 1.1])
    ax.set_ylim([1e-4, 1e-3])

    # Major and minor ticks on log scale
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Gridlines
    ax.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.5)

    # Tick parameters
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.tick_params(axis="both", which="minor", labelsize=7)


def plot_ber_grid(ax: plt.Axes):
    set_plot_style()
    ARRAY_SIZE = (4, 4)
    param_dict = initialize_dict(ARRAY_SIZE)
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)
        xloc_list.append(xloc)
        yloc_list.append(yloc)

    plot_parameter_array(
        ax,
        xloc_list,
        yloc_list,
        param_dict["bit_error_rate"],
        log=True,
        cmap=plt.get_cmap("Blues").reversed(),
    )

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cax = ax.inset_axes([1.10, 0, 0.1, 1])
    fig = plt.gcf()
    cbar = fig.colorbar(
        ax.get_children()[0], cax=cax, orientation="vertical", label="minimum BER"
    )

    return ax


def import_write_sweep_formatted() -> list[dict]:
    dict_list = import_directory(
        os.path.join(os.path.dirname(__file__), "write_current_sweep_enable_write/data")
    )
    dict_list = dict_list[1:]
    dict_list = dict_list[::-1]
    dict_list = sorted(
        dict_list, key=lambda x: x.get("enable_write_current").flatten()[0]
    )
    return dict_list


def import_delay_dict() -> dict:
    dict_list = import_directory(
        os.path.join(os.path.dirname(__file__), "read_delay_v2/data3")
    )
    delay_list = []
    bit_error_rate_list = []
    for data_dict in dict_list:
        delay = data_dict.get("delay").flatten()[0] * 1e-3
        bit_error_rate = get_bit_error_rate(data_dict)

        delay_list.append(delay)
        bit_error_rate_list.append(bit_error_rate)

    delay_dict = {}
    delay_dict["delay"] = delay_list
    delay_dict["bit_error_rate"] = bit_error_rate_list
    return delay_dict


def import_write_sweep_formatted_markers(dict_list) -> list[dict]:
    data = []
    data2 = []
    for data_dict in dict_list:
        bit_error_rate = get_bit_error_rate(data_dict)
        berargs = get_bit_error_rate_args(bit_error_rate)
        write_currents = get_read_currents(
            data_dict
        )  # This is correct. "y" is the write current in this .mat.
        enable_write_current = get_enable_write_current(data_dict)
        read_current = get_read_current(data_dict)
        write_current = get_write_current(data_dict)

        for i, arg in enumerate(berargs):
            if arg is not np.nan:

                if i == 0:
                    data.append(
                        {
                            "write_current": write_currents[arg],
                            "write_temp": get_channel_temperature(data_dict, "write"),
                            "read_current": read_current,
                            "enable_write_current": enable_write_current,
                        }
                    )
                if i == 2:
                    data2.append(
                        {
                            "write_current": write_currents[arg],
                            "write_temp": get_channel_temperature(data_dict, "write"),
                            "read_current": read_current,
                            "enable_write_current": enable_write_current,
                        }
                    )
    data_dict = {
        "data": data,
        "data2": data2,
    }
    return data_dict


def plot_fidelity_clean_bar(ber_array: np.ndarray, total_trials: int = 200_000) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # Prepare data
    barray = ber_array.copy().T
    ber_flat = barray.flatten()
    valid_mask = np.isfinite(ber_flat)
    ber_flat = ber_flat[valid_mask]
    fidelity_flat = 1 - ber_flat

    # Generate A1 to D4 labels
    rows = ["A", "B", "C", "D"]
    cols = range(1, 5)
    all_labels = [f"{r}{c}" for r in rows for c in cols]
    labels = np.array(all_labels)[valid_mask]

    # Error bars from binomial standard deviation
    errors = np.sqrt(ber_flat * (1 - ber_flat) / total_trials)
    # Clean label values
    display_values = [f"{f:.5f}" if f < 0.99999 else "≥0.99999" for f in fidelity_flat]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(len(fidelity_flat))
    ax.bar(x, fidelity_flat, yerr=errors, capsize=3, color="#658DDC", edgecolor="black")

    # Add value labels only if they fit in the visible range
    for i, (val, label) in enumerate(zip(fidelity_flat, display_values)):
        if val > 0.998:  # only plot label if within axis range
            ax.text(i, val + 1e-4, label, ha="center", va="bottom", fontsize=8)
        if val < 0.998:
            ax.text(i, 0.998 + 2e-4, label, ha="center", va="top", fontsize=8)
    # Formatting
    ax.set_xticks(x)
    ax.set_xlim(-1.5, len(x) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Fidelity (1 - BER)")
    ax.set_ylim(0.998, 1.0001)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Reference lines
    ax.axhline(0.999, color="#555555", linestyle="--", linewidth=0.8, zorder=3)
    ax.axhline(0.9999, color="#555555", linestyle="--", linewidth=0.8, zorder=3)
    # Labels drawn on top with optional white background
    # ax.text(
    #     len(x) + 0.3,
    #     0.99895,
    #     "0.999",
    #     va="top",
    #     ha="right",
    #     fontsize=10,
    #     color="k",
    #     zorder=4,
    #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    # )
    # ax.text(
    #     len(x) + 0.3,
    #     0.99985,
    #     "0.9999",
    #     va="top",
    #     ha="right",
    #     fontsize=10,
    #     color="k",
    #     zorder=4,
    #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    # )
    ax.axhline(
        1-1.5e-3,
        color="red",
        linestyle="--",
        linewidth=0.8,
        zorder=3,
    )
    ax.text(
        len(x) + 0.3,
        1-1.5e-3,
        "Previous Record",
        va="top",
        ha="right",
        fontsize=14,
        color="red",
        zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
    )
    ax.set_ylim(0.998, 1.0004)
    ax.set_yticks([0.998, 0.999, 0.9999])
    ax.set_yticklabels(["0.998", "0.999", "0.9999"])
    fig.patch.set_visible(False)
    save_fig = False
    if save_fig:
        plt.savefig(
            "fidelity_clean_bar.pdf",
            dpi=300,
            bbox_inches="tight",
        )
    plt.tight_layout()
    plt.show()


def plot_ber_3d_bar(ber_array: np.ndarray, total_trials: int = 200_000) -> None:
    barray = ber_array.copy()
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 4x4 grid
    x_data, y_data = np.meshgrid(np.arange(4), np.arange(4))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    dz = barray.flatten()

    # Mask invalid entries
    valid_mask = np.isfinite(dz) & (dz < 5.5e-2)
    x_data, y_data, dz = x_data[valid_mask], y_data[valid_mask], dz[valid_mask]

    # Compute error counts
    error_counts = np.round(dz * total_trials).astype(int)
    z_data = np.zeros_like(error_counts)
    dx = dy = 0.6 * np.ones_like(error_counts)
    print(error_counts)
    # Normalize error counts for colormap
    norm = Normalize(vmin=error_counts.min(), vmax=error_counts.max())
    colors = cm.Blues(norm(error_counts))

    # Plot 3D bars with color
    ax.bar3d(
        x_data,
        y_data,
        z_data,
        dx,
        dy,
        error_counts,
        shade=True,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )

    # Invert y-axis for logical row order
    ax.invert_yaxis()

    # Labels and ticks
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_zlabel("Errors")
    ax.set_title("Error Count per Cell")
    ax.set_xticks(np.arange(4) + 0.3)  # Offset tick locations
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_yticks(np.arange(4) + 0.3)  # Offset tick locations
    ax.set_yticklabels(["1", "2", "3", "4"])
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_zlim(1, 500)
    ax.view_init(elev=45, azim=220)

    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Errors (per 200k)")
    fig.tight_layout()
    fig.patch.set_visible(False)
    save_fig = False
    if save_fig:
        fig.savefig("ber_3d_bar.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    param_dict = initialize_dict((4, 4))
    xloc_list = []
    yloc_list = []
    for c in CELLS:
        xloc, yloc = convert_cell_to_coordinates(c)
        param_dict = process_cell(CELLS[c], param_dict, xloc, yloc)

    ber_array = param_dict["bit_error_rate"]
    valid_ber = ber_array[np.isfinite(ber_array) & (ber_array < 5.5e-2)]

    average_ber = np.mean(valid_ber)
    std_ber = np.std(valid_ber)
    min_ber = np.min(valid_ber)
    max_ber = np.max(valid_ber)
    print(len(valid_ber))
    print("=== Array BER Statistics ===")
    print(f"Average BER: {average_ber:.2e}")
    print(f"Std Dev BER: {std_ber:.2e}")
    print(f"Min BER: {min_ber:.2e}")
    print(f"Max BER: {max_ber:.2e}")
    print("=============================")

    # Plot the 3D bar chart
    plot_ber_3d_bar(ber_array)

    plot_fidelity_clean_bar(ber_array)
