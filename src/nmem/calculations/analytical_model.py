import matplotlib.pyplot as plt
import numpy as np

from nmem.calculations.calculations import (
    calculate_persistent_currents,
    calculate_read_currents,
    htron_critical_current,
)
from nmem.calculations.plotting import (
    plot_persistent_current,
    plot_read_current,
)
from nmem.measurement.cells import CELLS


def create_data_dict(
    enable_read_currents: np.ndarray,
    read_currents: np.ndarray,
    width_left: float,
    width_right: float,
    alpha: float,
    iretrap_enable: float,
    max_critical_current: float,
    htron_slope: float,
    htron_intercept: float,
    persistent_current: float = 0.0,
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
        "persistent_current": persistent_current,
    }

    return data_dict


if __name__ == "__main__":
    current_cell = "C1"
    HTRON_SLOPE = CELLS[current_cell]["slope"]
    HTRON_INTERCEPT = CELLS[current_cell]["y_intercept"]
    WIDTH_LEFT = 0.1
    WIDTH_RIGHT = 0.213
    ALPHA = 0.563
    PERSISTENT_CURRENT = 30.0
    MAX_CRITICAL_CURRENT = 860  # CELLS[current_cell]["max_critical_current"]
    IRETRAP_ENABLE = 0.573
    IREAD = 630
    N = 260

    enable_read_currents = np.linspace(0, 400, N)
    read_currents = np.linspace(00, 1050, N)

    data_dict = create_data_dict(
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

    fig, ax = plt.subplots()
    persistent_currents_mesh = calculate_persistent_currents(data_dict)
    plot_persistent_current(
        ax,
        left_critical_currents_mesh=data_dict["left_critical_currents_mesh"],
        write_currents_mesh=data_dict["write_currents_mesh"],
        total_persistent_current=persistent_currents_mesh,
        width_ratio=data_dict["width_ratio"],
    )

    data_dict["persistent_current"] = PERSISTENT_CURRENT

    fig, ax = plt.subplots()
    read_current_dict = calculate_read_currents(data_dict)
    data_dict.update(read_current_dict)
    plot_read_current(ax, data_dict)
    # read_current_dict = calculate_read_currents(data_dict)

    ax.plot(data_dict["channel_critical_currents"], data_dict["fa"], label="fa")
    ax.plot(data_dict["channel_critical_currents"], data_dict["fb"], label="fb")
    ax.plot(data_dict["channel_critical_currents"], data_dict["fc"], label="fc")
