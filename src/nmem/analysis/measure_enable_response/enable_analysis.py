import matplotlib.pyplot as plt
import numpy as np
from nmem.analysis.analysis import import_directory, plot_enable_current_relation
from nmem.measurement.functions import (
    plot_fitting,
    build_array,
    filter_plateau,
    get_fitting_points,
    calculate_channel_temperature,
)
import os
import scipy.io as sio
from nmem.measurement.cells import CELLS


def plot_grid(axs, dict_list):
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for dict in dict_list:
        cell = dict.get("cell")[0]

        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1
        x = dict["x"][0]
        y = dict["y"][0]
        ztotal = dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)
        # xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.9)
        axs[row, column].plot(
            xfit, yfit, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )

        xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.75)
        plot_fitting(
            axs[row, column],
            xfit,
            yfit,
            color=colors[column],
            marker=markers[row],
            markeredgecolor="k",
        )
        enable_read_current = CELLS[cell].get("enable_read_current")*1e6
        enable_write_current = CELLS[cell].get("enable_write_current")*1e6
        axs[row, column].vlines(
            [enable_write_current],
            *axs[row, column].get_ylim(),
            linestyle="--",
            color="grey",
            label="Enable Write Current",
        )
        axs[row, column].vlines(
            [enable_read_current],
            *axs[row, column].get_ylim(),
            linestyle="--",
            color="r",
            label="Enable Read Current",
        )

        
        axs[row, column].legend(loc="upper right")
        axs[row, column].set_xlim(0, 600)
        axs[row, column].set_ylim(0, 1000)
        # axs[row, column].set_aspect("equal")
    axs[-1, 0].set_xlabel("Enable Current ($\mu$A)")
    axs[-1, 0].set_ylabel("Critical Current ($\mu$A)")
    return axs


def plot_row(axs, dict_list):
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for dict in dict_list:
        cell = dict.get("cell")[0]

        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1
        x = dict["x"][0]
        y = dict["y"][0]
        ztotal = dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)
        # xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.9)
        axs[row].plot(
            xfit, yfit, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )

        axs[row].legend(loc="lower left")
        axs[row].set_xlim(0, 500)
        axs[row].set_ylim(0, 1000)
    axs[0].set_xlabel("Enable Current ($\mu$A)")
    axs[0].set_ylabel("Critical Current ($\mu$A)")
    return axs


def plot_column(axs, dict_list):
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for dict in dict_list:
        cell = dict.get("cell")[0]

        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1
        x = dict["x"][0]
        y = dict["y"][0]
        ztotal = dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)
        # xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.9)
        axs[column].plot(
            xfit, yfit, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )

        axs[column].legend(loc="lower left")
        axs[column].set_xlim(0, 500)
        axs[column].set_ylim(0, 1000)
    axs[0].set_xlabel("Enable Current ($\mu$A)")
    axs[0].set_ylabel("Critical Current ($\mu$A)")
    return axs


def plot_full_grid():

    dict_list = import_directory("data")
    fig, axs = plt.subplots(5, 5, figsize=(20, 20), sharex=True, sharey=True)

    plot_grid(axs[1:5, 0:4], dict_list)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    plot_row(axs[0, 0:4], dict_list)

    plot_column(axs[1:5, 4], dict_list)
    axs[0, 4].axis("off")
    axs[4, 0].set_xlabel("Enable Current ($\mu$A)")
    axs[4, 0].set_ylabel("Critical Current ($\mu$A)")

    plt.show()


def plot_all_cells():
    dict_list = import_directory("data")
    fig, axs = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    avg_slope, avg_intercept = get_average_response(CELLS)

    for dict in dict_list:
        cell = dict.get("cell")[0]

        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1
        x = dict["x"][0]
        y = dict["y"][0]
        ztotal = dict["ztotal"]
        xfit, yfit = get_fitting_points(x, y, ztotal)
        # xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.9)
        axs.plot(xfit, yfit, label=f"{cell}", color=colors[column], marker=markers[row])

        xfit, yfit = filter_plateau(xfit, yfit, yfit[0] * 0.75)
        # plot_fitting(axs, xfit, yfit, color="k")

        x = np.linspace(0, -avg_intercept / avg_slope, 100)
        y = avg_slope * x + avg_intercept
        axs.plot(x, y, color="k", linestyle="--")
        axs.plot()
        axs.set_xlabel("Enable Current ($\mu$A)")
        axs.set_ylabel("Critical Current ($\mu$A)")
        axs.legend(loc="lower left", ncol=4)


def get_average_response(cell_dict):
    slope_list = []
    intercept_list = []
    for value in cell_dict.values():
        slope_list.append(value.get("slope"))
        intercept_list.append(value.get("y_intercept"))

    slope = np.mean(slope_list)
    intercept = np.mean(intercept_list)
    return slope, intercept


def plot_enable_write_temperature():
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for cell in CELLS.keys():
        xint = CELLS[cell].get("x_intercept")
        x = CELLS[cell].get("enable_write_current") * 1e6
        temp = calculate_channel_temperature(1.3, 12.3, x, xint)
        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1
        ax.plot(
            x, temp, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Enable Write Current ($\mu$A)")
    ax.set_ylabel("Channel Temperature (K)")
    ax.set_ylim(0, 13)
    ax.hlines(12.3, 0, 500, linestyle="--")


def plot_enable_read_temperature():
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    markers = ["o", "s", "D", "^"]
    for cell in CELLS.keys():
        xint = CELLS[cell].get("x_intercept")
        x = CELLS[cell].get("enable_read_current") * 1e6
        temp = calculate_channel_temperature(1.3, 12.3, x, xint)
        column = ord(cell[0]) - ord("A")
        row = int(cell[1]) - 1
        ax.plot(
            x, temp, label=f"Cell {cell}", color=colors[column], marker=markers[row]
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("Enable Read Current ($\mu$A)")
    ax.set_ylabel("Channel Temperature (K)")
    ax.set_ylim(0, 13)
    ax.hlines(12.3, 0, 500, linestyle="--")


if __name__ == "__main__":
    # plot_full_grid()
    # plot_all_cells()

    # dict_list = import_directory("data")
    # fig, ax = plt.subplots(1, 4, figsize=(12, 6), sharey=True)
    # plot_column(ax, dict_list)
    # plt.show()

    dict_list = import_directory("data")
    fig, axs = plt.subplots(4,4, figsize=(20, 20), sharex=True, sharey=True)
    plot_grid(axs, dict_list)

    