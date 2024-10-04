import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from nmem.calculations.calculations import (
    calculate_persistent_current,
    calculate_read_currents,
    htron_critical_current,
)
from nmem.calculations.plotting import (
    plot_htron_sweep,
    plot_persistent_current,
    plot_read_current,
)
from nmem.measurement.cells import CELLS


def import_matlab_data(file_path):
    data = sio.loadmat(file_path)
    return data


def persistent_current_plot(data_dict):
    persistent_current, regions = calculate_persistent_current(data_dict)

    data_dict["persistent_currents"] = persistent_current
    data_dict["regions"] = regions

    # Plot the persistent current
    fig, ax = plt.subplots()
    ax, _ = plot_persistent_current(
        ax,
        data_dict,
        plot_regions=False,
    )


def read_current_plot(data_dict, persistent_current=None):
    persistent_currents, regions = calculate_persistent_current(data_dict)
    data_dict["regions"] = regions
    if persistent_current == 0:
        data_dict["persistent_currents"] = np.zeros_like(persistent_currents)
    else:
        data_dict["persistent_currents"] = (
            np.ones_like(persistent_currents) * persistent_current
        )

    fig, ax = plt.subplots()
    ax, read_current_dict = plot_read_current(
        ax, data_dict, contour=False, plot_regions=False
    )
    ax.set_aspect("equal")
    plt.hlines(
        data_dict["max_critical_current"] * 1e6,
        ax.get_xlim()[0],
        ax.get_xlim()[1],
        color="red",
        linestyle="--",
    )

    plt.text(
        0.15,
        0.75,
        f"$I_{{P}}$={persistent_current}$\mu$A",
        fontsize=24,
        color="black",
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    plt.plot([860, 860, 860], [840, 727, 655], marker="o", color="red", ls="")
    plt.xlim(600, 950)
    plt.ylim(500, 950)
    return data_dict




def plot_experimental_data():
    FILE_PATH = "/home/omedeiro/"
    FILE_NAME = "SPG806_20240804_nMem_ICE_ber_D6_A4_2024-08-04 16-38-50.mat"
    matlab_data_dict = import_matlab_data(FILE_PATH + "/" + FILE_NAME)
    # Extract write enable sweep data.
    enable_write_currents_measured = np.transpose(matlab_data_dict["x"][:, :, 1]) * 1e6
    write_currents_measured = np.transpose(matlab_data_dict["y"][:, :, 1]) * 1e6
    ber = matlab_data_dict["ber"]
    ber_2D = np.reshape(
        ber,
        (len(write_currents_measured), len(enable_write_currents_measured)),
        order="F",
    )
    # Plot experimental data
    fig, ax = plt.subplots()
    ax = plot_htron_sweep(
        ax, write_currents_measured, enable_write_currents_measured, ber_2D
    )


def create_dict_read(
    enable_read_currents,
    read_currents,
    width_left,
    width_right,
    alpha,
    iretrap_enable,
    max_critical_current,
    htron_slope,
    htron_intercept,
):
    width_ratio = width_right / width_left

    # Calculate the channel critical current
    channel_critical_currents = htron_critical_current(
        enable_read_currents,
        htron_slope,
        htron_intercept,
    )

    [channel_critical_currents_mesh, read_currents_mesh] = np.meshgrid(
        channel_critical_currents, read_currents
    )
    write_currents_mesh = read_currents_mesh

    right_critical_currents = channel_critical_currents / (
        1 + (iretrap_enable / width_ratio)
    )
    left_critical_currents = right_critical_currents / width_ratio
    [left_critical_currents_mesh, read_currents_mesh] = np.meshgrid(
        left_critical_currents, read_currents
    )
    [right_critical_currents_mesh, read_currents_mesh] = np.meshgrid(
        right_critical_currents, read_currents
    )
    # Create the data dictionary
    data_dict = {
        "left_critical_currents": left_critical_currents,
        "right_critical_currents": right_critical_currents,
        "left_critical_currents_mesh": left_critical_currents_mesh,
        "right_critical_currents_mesh": right_critical_currents_mesh,
        "channel_critical_currents": channel_critical_currents,
        "channel_critical_currents_mesh": channel_critical_currents_mesh,
        "read_currents_mesh": read_currents_mesh,
        "write_currents_mesh": write_currents_mesh,
        "alpha": alpha,
        "iretrap_enable": iretrap_enable,
        "width_left": width_left,
        "width_right": width_right,
        "width_ratio": width_ratio,
        "max_critical_current": max_critical_current,
    }

    return data_dict


if __name__ == "__main__":
    current_cell = "C1"
    HTRON_SLOPE = CELLS[current_cell]["slope"]
    HTRON_INTERCEPT = CELLS[current_cell]["intercept"]
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.192
    ALPHA = 0.655

    MAX_CRITICAL_CURRENT = 860e-6  # CELLS[current_cell]["max_critical_current"]
    IRETRAP_ENABLE = 0.625
    IREAD = 630
    N = 200

    enable_read_currents = np.linspace(0, 400, N)
    read_currents = np.linspace(400, 1050, N)

    data_dict = create_dict_read(
        enable_read_currents,
        read_currents,
        WIDTH_LEFT,
        WIDTH_RIGHT,
        ALPHA,
        IRETRAP_ENABLE,
        MAX_CRITICAL_CURRENT,
        HTRON_SLOPE,
        HTRON_INTERCEPT,
    )

    # persistent_current_plot(data_dict)
    data_dict = read_current_plot(data_dict, persistent_current=30)

    read_current_dict = calculate_read_currents(data_dict)
