import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from nmem.calculations.calculations import (
    calculate_ideal_read_current,
    calculate_ideal_read_margin,
    calculate_left_branch_current,
    calculate_left_lower_bound,
    calculate_left_upper_bound,
    calculate_one_state_current,
    calculate_right_branch_current,
    calculate_right_lower_bound,
    calculate_right_upper_bound,
    calculate_zero_state_current,
    htron_critical_current,
)
from nmem.calculations.plotting import (
    plot_htron_sweep,
    plot_persistent_current,
    plot_read_current,
    plot_read_margin,
    plot_state_currents,
)
from nmem.measurement.cells import CELLS


def import_matlab_data(file_path):
    data = sio.loadmat(file_path)
    return data


def get_point_parameters(i: int, j: int, data_dict: dict):
    left_critical_current = float(data_dict["left_critical_currents"][i])
    right_critical_current = float(data_dict["right_critical_currents"][i])
    write_current = float(data_dict["write_currents"][j])
    enable_write_current = float(data_dict["enable_write_currents"][i])
    persistent_currents = data_dict["persistent_currents"]
    persistent_current = float(data_dict["persistent_currents"][j, i])
    max_left_critical_current = data_dict["max_left_critical_current"]
    max_right_critical_current = data_dict["max_right_critical_current"]
    iretrap = data_dict["iretrap"]
    alpha = data_dict["alpha"]
    width_ratio = data_dict["width_ratio"]
    set_read_current = data_dict["set_read_current"]
    max_critical_current = data_dict["max_critical_current"]
    # Initial Write
    left_branch_write_current = calculate_left_branch_current(alpha, write_current, 0)
    right_branch_write_current = calculate_right_branch_current(alpha, write_current, 0)

    # Set Read
    zero_state_current = calculate_zero_state_current(
        left_critical_current,
        right_critical_current,
        persistent_current,
        alpha,
        iretrap,
        max_critical_current,
    )
    one_state_current = calculate_one_state_current(
        left_critical_current,
        right_critical_current,
        persistent_current,
        alpha,
        iretrap,
        max_critical_current,
    )

    ideal_read_current = calculate_ideal_read_current(
        zero_state_current, one_state_current
    )
    ideal_read_margin = calculate_ideal_read_margin(
        zero_state_current, one_state_current
    )

    noninverting_operation = zero_state_current < one_state_current

    min_ichl = calculate_left_lower_bound(persistent_current, set_read_current, alpha)
    max_ichl = calculate_left_upper_bound(persistent_current, set_read_current, alpha)
    min_ichr = calculate_right_lower_bound(persistent_current, set_read_current, alpha)
    max_ichr = calculate_right_upper_bound(persistent_current, set_read_current, alpha)

    ichl_within_range = min_ichl < left_critical_current < max_ichl
    ichr_within_range = min_ichr < right_critical_current < max_ichr

    set_param_dict = {
        "Total Critical Current (enable off) [uA]": f"{HTRON_INTERCEPT:.2f}",
        "Left Critical Current (enable off) [uA]": f"{max_left_critical_current:.2f}",
        "Right Critical Current (enable off) [uA]": f"{max_right_critical_current:.2f}",
        "Width Ratio": f"{width_ratio:.2f}",
        "Inductive Ratio (alpha)": f"{alpha:.2f}",
        "Retrap Ratio": f"{iretrap:.2f}",
    }
    write_param_dict = {
        "Write Current [uA]": f"{write_current:.2f}",
        "Enable Write Current [uA]": f"{enable_write_current:.2f}",
        "Left Branch Write Current [uA]": f"{left_branch_write_current:.2f}",
        "Right Branch Write Current [uA]": f"{right_branch_write_current:.2f}",
        "Left Side Critical Current (enable on) [uA]": f"{left_critical_current:.2f}",
        "Right Side Critical Current (enable on) [uA]": f"{right_critical_current:.2f}",
        "Left Side Retrapping Current (enable on) [uA]": f"{left_critical_current*iretrap:.2f}",
        "Right Side Retrapping Current (enable on) [uA]": f"{right_critical_current*iretrap:.2f}",
        "Maximum Persistent Current": f"{persistent_current:.2f}",
    }
    read_param_dict = {
        "Left Side Critical Current (enable on) [uA]": f"{left_critical_current:.2f}",
        "Right Side Critical Current (enable on) [uA]": f"{right_critical_current:.2f}",
        "Maximum Persistent Current": f"{persistent_current:.2f}",
        "Zero State Current [uA]": f"{zero_state_current:.2f}",
        "One State Current [uA]": f"{one_state_current:.2f}",
        "Set Read Current [uA]": f"{set_read_current:.2f}",
        "Ideal Read Current [uA]": f"{ideal_read_current:.2f}",
        "Ideal Read Margin [uA]": f"{ideal_read_margin:.2f}",
        "Non-inverting Operation": f"{noninverting_operation}",
        "ICHL within range": f"{ichl_within_range}",
        "ICHR within range": f"{ichr_within_range}",
        "minimum ICHL [uA] (set read)": f"{min_ichl:.2f}",
        "maximum ICHL [uA] (set read)": f"{max_ichl:.2f}",
        "minimum ICHR [uA] (set read)": f"{min_ichr:.2f}",
        "maximum ICHR [uA] (set read)": f"{max_ichr:.2f}",
    }
    param_dict = {
        # **set_param_dict,
        # **write_param_dict,
        **read_param_dict,
    }

    param_df = pd.DataFrame(param_dict.values(), index=param_dict.keys())
    param_df.columns = ["Value"]

    return param_df


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


if __name__ == "__main__":
    current_cell = "C1"
    HTRON_SLOPE = CELLS[current_cell]["slope"]
    HTRON_INTERCEPT = CELLS[current_cell]["intercept"]
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.3
    ALPHA = 1 - 0.3

    MAX_CRITICAL_CURRENT = CELLS[current_cell]["max_critical_current"]
    IRETRAP = 0.2
    IREAD = 200
    IDXX = 30
    IDXY = 35
    N = 100

    width_ratio = WIDTH_RIGHT / WIDTH_LEFT
    max_left_critical_current = HTRON_INTERCEPT / width_ratio
    max_right_critical_current = HTRON_INTERCEPT

    # plot_experimental_data()

    # Define the write and enable write currents
    enable_write_currents = np.linspace(300, 450, N)
    write_currents = np.linspace(00, 550, N)

    # Calculate the channel critical current
    channel_current_enabled = htron_critical_current(
        enable_write_currents,
        HTRON_SLOPE,
        HTRON_INTERCEPT,
    )
    # Define the critical currents for the left and right branches
    left_critical_currents = channel_current_enabled / width_ratio
    right_critical_currents = channel_current_enabled

    # Create the meshgrid for the critical currents
    [left_critical_currents_mesh, write_currents_mesh] = np.meshgrid(
        left_critical_currents, write_currents
    )
    right_critical_currents_mesh = left_critical_currents_mesh * width_ratio

    # Create the data dictionary
    data_dict = {
        "left_critical_currents": left_critical_currents,
        "right_critical_currents": right_critical_currents,
        "left_critical_currents_mesh": left_critical_currents_mesh,
        "right_critical_currents_mesh": right_critical_currents_mesh,
        "write_currents": write_currents,
        "write_currents_mesh": write_currents_mesh,
        "enable_write_currents": enable_write_currents,
        "alpha": ALPHA,
        "iretrap": IRETRAP,
        "set_read_current": IREAD,
        "width_left": WIDTH_LEFT,
        "width_right": WIDTH_RIGHT,
        "width_ratio": width_ratio,
        "max_channel_critical_current": HTRON_INTERCEPT,
        "max_critical_current": MAX_CRITICAL_CURRENT,
        "max_left_critical_current": max_left_critical_current,
        "max_right_critical_current": max_right_critical_current,
    }

    # Plot the persistent current
    fig, ax = plt.subplots()
    ax, data_dict = plot_persistent_current(
        ax,
        data_dict,
        plot_regions=False,
        # data_point=(left_critical_currents[IDXX], write_currents[IDXY]),
    )

    fig, ax = plt.subplots()
    ax, data_dict = plot_read_current(ax, data_dict, contour=True)

    zero_state_currents = calculate_zero_state_current(
        left_critical_currents_mesh,
        right_critical_currents_mesh,
        data_dict["persistent_currents"],
        ALPHA,
        IRETRAP,
        CELLS[current_cell]["max_critical_current"],
        width_ratio,
    )
    one_state_currents = calculate_one_state_current(
        left_critical_currents_mesh,
        right_critical_currents_mesh,
        data_dict["persistent_currents"],
        ALPHA,
        IRETRAP,
        CELLS[current_cell]["max_critical_current"],
        width_ratio,
    )

    data_dict["zero_state_currents"] = zero_state_currents
    data_dict["one_state_currents"] = one_state_currents

    data_dict["ideal_read_margins"] = calculate_ideal_read_margin(
        zero_state_currents, one_state_currents
    )

    # Plot the read margin
    fig, ax = plt.subplots()
    read_margin = plot_read_margin(ax, data_dict)

    # # Get the point parameters
    # param_df = get_point_parameters(IDXX, IDXY, data_dict)
    # print(param_df)

    plt.show()

    plot_state_currents(data_dict)